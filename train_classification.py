import sys
import os
import numpy as np
import yaml
from collections import defaultdict
import torch
from torch import nn
import tqdm
import argparse
sys.path.append(os.path.realpath(__file__))

from datasets.scanobjectnn import ScanObjectNN

from utils import train_util
from utils import train_util_distributed

import torch.distributed as dist
import torch.optim

import torch.utils.data
import torch.utils.data.distributed


torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="path to py model", required=True, default='.configs/config_seg.yaml')

parser.add_argument("exp_name", help="name of the exp")


parser.add_argument("--master", required=True)
parser.add_argument("--rank", required=True, type=int)
parser.add_argument("--num_nodes", required=True, type=int)


args = parser.parse_args()
config_path = args.config


dist_backend = 'nccl'
# Url used to setup distributed training
dist_url = "tcp://{}".format(args.master)


dist.init_process_group(backend=dist_backend,
                        init_method=dist_url,
                        rank=args.rank,
                        world_size=args.num_nodes)


with open(config_path, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

# check if we masked all other devices
assert(torch.cuda.device_count() == 1)
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

gen_model_file, gen_model_name = train_util.check_model_paths(cfg['model']['generator'])[0]


exp_descr = args.exp_name

scan_train = ScanObjectNN(data_dir=cfg['data']['path'],
                          train=True,
                          center=cfg['data']['center'],
                          normalize=cfg['data']['normalize'],
                          subsample=cfg['data']['subsample'] if 'subsample' in cfg['data'] else None)

scan_val = ScanObjectNN(data_dir=cfg['data']['path_val'],
                        train=False,
                        center=cfg['data']['center'],
                        normalize=cfg['data']['normalize'],
                        subsample=cfg['data']['subsample'] if 'subsample' in cfg['data'] else None)

train_sampler = torch.utils.data.distributed.DistributedSampler(scan_train)
val_sampler = torch.utils.data.distributed.DistributedSampler(scan_val)

dataloader_train = torch.utils.data.DataLoader(scan_train,
                                               batch_size=cfg['data']['batch_size'],
                                               shuffle=(train_sampler is None),
                                               num_workers=cfg['data']['num_workers'],
                                               pin_memory=False, sampler=train_sampler)

dataloader_val = torch.utils.data.DataLoader(scan_val,
                                             batch_size=cfg['data']['batch_size_val'],
                                             shuffle=False,
                                             num_workers=cfg['data']['num_workers'],
                                             pin_memory=False, sampler=val_sampler)

# model part
del cfg['model']['generator']
params_dict_gen = cfg['model']

torch.manual_seed(42)
if args.rank == 0:
    writer, exp_dir, full_desc = train_util.create_experiment(exp_descr,
                                                              params_dict={'exp_root': cfg['experiment']['root'],
                                                         'writer_root': cfg['experiment']['writer_root'],
                                                                           'config_path': config_path})
    generator = train_util.get_model(gen_model_file, params_dict_gen,
                                     exp_dir=exp_dir).cuda()
else:
    generator = train_util.get_model(gen_model_file, params_dict_gen, exp_dir=None).cuda()

generator = torch.nn.parallel.DistributedDataParallel(torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator),
                                                      device_ids=[0],
                                                      output_device=0)

# print('generator', generator)
print('generator_params', sum(p.numel() for p in generator.parameters()))

if 'scale_lr' in cfg['train']:
    opt_params_dict = [{'params': map(lambda p: p[1],
                                      filter(lambda p: not p[0].endswith('scale'),
                                             generator.named_parameters()))},
                       {'params': map(lambda p: p[1],
                                      filter(lambda p: p[0].endswith('scale'),
                                             generator.named_parameters())), 'lr': cfg['train']['scale_lr']}]
else:
    opt_params_dict = generator.parameters()

optimizer = train_util.make_optimizer(opt_params_dict, cfg['train']['optimizer'])

scheduler_adaptive = cfg['train']['scheduler']['type'] == 'ReduceLROnPlateau'
scheduler = train_util.make_scheduler(optimizer, cfg['train']['scheduler'])

# print(scheduler)
# print('Is scheduler adaptive? {}'.format(scheduler_adaptive))

if 'restore' in cfg:
    print('restoring')
    g_ckpt_path = cfg['restore']['generator']
    g_opt_ckpt_path = cfg['restore']['optimizer']
    train_util_distributed.restore_exp(objects=[generator, optimizer],
                                       names=[g_ckpt_path, g_opt_ckpt_path])

    if 'new_lr' in cfg['restore']:
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg['restore']['new_lr']

show_each = cfg['train']['show_each']
data_iters = 0

seg_loss_weight = cfg['train']['seg_weight']

label_smooth = False

if 'label_smooth' in cfg['train']:
    label_smooth = cfg['train']['label_smooth']

cross_entropy_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCEWithLogitsLoss()


def gather_results(loss_dict,
                   class_pred, mask_pred, labels, mask):
    loss_all = train_util_distributed.reduce_loss_dict(loss_dict)

    with torch.no_grad():
        class_pred_this = np.argmax(class_pred.detach().cpu().numpy(), axis=1)

        mask_pred_this = (torch.sigmoid(mask_pred[:, 0, 0]) > 0.5).detach().cpu().numpy()

        mask_this = mask.detach().cpu().numpy()
        labels_this = labels.detach().cpu().numpy()

    all_gathered = train_util_distributed.all_gather((class_pred_this,
                                                      mask_pred_this,
                                                      labels_this,
                                                      mask_this))

    return loss_all, all_gathered

n_classes = 15

max_val_acc = 0
max_val_macc = 0

for epoch in range(cfg['train']['num_epochs']):
    train_sampler.set_epoch(epoch)
    generator.train()

    total_correct = 0
    total_correct_seg = 0
    total_seen = 0
    total_seen_seg = 0

    loss_train_all = []
    cls_loss_train_all = []
    seg_loss_train_all = []

    for i, (pcd, labels, mask) in tqdm.tqdm(enumerate(dataloader_train)):
        pcd = pcd.permute(0, 2, 1)[:, :, None].cuda()
        mask = mask.float().cuda(non_blocking=True)
        labels = labels.long().cuda(non_blocking=True)

        class_pred, mask_pred, lattices_sizes = generator(pcd)

        seg_loss = bce_loss(mask_pred[:, 0, 0], mask)
        cls_loss = cross_entropy_loss(class_pred, labels)

        loss = (1-seg_loss_weight) * cls_loss + seg_loss_weight * seg_loss

        loss.backward()

        if 'grad_stats' in cfg['train']:
            with torch.no_grad():
                if args.rank == 0 and data_iters % cfg['train']['grad_stats']['iters'] == 0:
                    for name, param in generator.module.named_parameters():
                        # writer.add_histogram('weight_' + name, param, global_step=data_iters)
                        if param.requires_grad:
                            if param.grad is None:
                                print(name)

                            if cfg['train']['grad_stats']['hist']:
                                writer.add_histogram('stats/grad_' + name,
                                                     param.grad,
                                                     global_step=data_iters, bins='auto')
                            writer.add_scalar('stats/grad_n_' + name,
                                              torch.norm(param.grad),
                                              global_step=data_iters)

                    writer.flush()

        optimizer.step()
        optimizer.zero_grad()

        loss_all, all_gathered = gather_results(loss_dict={'loss': loss, 'loss_cls': cls_loss,
                                                            'loss_seg': seg_loss},
                                                class_pred=class_pred,
                                                mask_pred=mask_pred,
                                                labels=labels,
                                                mask=mask)

        if args.rank == 0:
            loss_train_all.append(loss_all['loss'].item())
            cls_loss_train_all.append(loss_all['loss_cls'].item())
            seg_loss_train_all.append(loss_all['loss_seg'].item())

            for class_pred_np, mask_pred_np, labels_np, mask_np in all_gathered:
                total_correct += np.sum(class_pred_np == labels_np)
                total_seen += labels_np.shape[0]

                total_correct_seg += np.sum(mask_pred_np == mask_np)
                total_seen_seg += mask_np.shape[0] * mask_np.shape[-1]

        if args.rank == 0:
            for key in loss_all.keys():
                writer.add_scalar('train/{}'.format(key), loss_all[key].item(), global_step=data_iters)

            # lattice stats
            for i, value in enumerate(lattices_sizes):
                writer.add_scalar('train/lattice_{}'.format(i),
                                  value[0], global_step=data_iters)
                writer.add_scalar('train/norm_l_feat_{}'.format(i),
                                  value[1].item(), global_step=data_iters)
                writer.add_scalar('train/norm_l_feat_var_{}'.format(i),
                                  value[2].item(), global_step=data_iters)

            if data_iters % cfg['train']['save_each'] == 0 and data_iters > 0:
                generator.eval()

                train_util_distributed.save_exp_parallel([generator, optimizer],
                                                         ['generator', 'g_opt'], exp_path=exp_dir,
                                                         epoch=data_iters, epoch_name='iter')
                generator.train()

        data_iters += 1

        if not scheduler_adaptive:
            scheduler.step(data_iters)

        del pcd, labels

    if args.rank == 0:
        print('on train cls_acc {}, seg_acc {}'.format(total_correct / float(total_seen),
                                                       total_correct_seg / float(total_seen_seg)))

    if args.rank == 0 and epoch % cfg['train']['save_each_epoch'] == 0 and epoch > 0:
        print('saving ckpt')
        train_util.save_exp([generator, optimizer],
                            ['generator', 'g_opt'], exp_path=exp_dir, epoch=epoch, epoch_name='epoch')

    if epoch % cfg['train']['val_step'] == 0:
        all_loss = []
        all_loss_seg = []
        all_loss_cls = []

        total_correct = 0
        total_correct_seg = 0
        total_seen = 0
        total_seen_seg = 0

        correct_per_label = np.zeros(n_classes)
        total_per_label = np.zeros(n_classes)

        generator.eval()
        val_sampler.set_epoch(epoch)

        lattice_info = defaultdict(list)

        with torch.no_grad():
            for i, (pcd, labels, mask) in tqdm.tqdm(enumerate(dataloader_val)):
                pcd = pcd.permute(0, 2, 1)[:, :, None].cuda()
                mask = mask.float().cuda(non_blocking=True)
                labels = labels.long().cuda(non_blocking=True)

                class_pred, mask_pred, lattices_sizes = generator(pcd)

                seg_loss = bce_loss(mask_pred[:, 0, 0], mask)
                cls_loss = cross_entropy_loss(class_pred, labels)

                loss = (1 - seg_loss_weight) * cls_loss + seg_loss_weight * seg_loss

                for ind, value in enumerate(lattices_sizes):
                    lattice_info['lattice_{}'.format(ind)].append(value[0])
                    lattice_info['norm_l_feat_{}'.format(ind)].append(value[1].item())
                    lattice_info['norm_l_feat_var_{}'.format(ind)].append(value[2].item())

                loss_all, all_gathered = gather_results(loss_dict={'loss': loss, 'loss_cls': cls_loss,
                                                                   'loss_seg': seg_loss},
                                                        class_pred=class_pred,
                                                        mask_pred=mask_pred,
                                                        labels=labels,
                                                        mask=mask)

                all_loss.append(loss_all['loss'].item())
                all_loss_cls.append(loss_all['loss_cls'].item())
                all_loss_seg.append(loss_all['loss_seg'].item())

                if args.rank == 0:
                    for class_pred_np, mask_pred_np, labels_np, mask_np in all_gathered:
                        total_correct += np.sum(class_pred_np == labels_np)
                        total_seen += labels_np.shape[0]

                        for batch_id in range(labels_np.shape[0]):
                            correct_per_label[labels_np[batch_id]] += class_pred_np[batch_id] == labels_np[batch_id]
                            total_per_label[labels_np[batch_id]] += 1

                        total_correct_seg += np.sum(mask_pred_np == mask_np)
                        total_seen_seg += mask_np.shape[0] * mask_np.shape[-1]

                del pcd, labels, mask

        if args.rank == 0:
            writer.add_scalar('val/cls_acc', total_correct / float(total_seen), global_step=epoch)
            writer.add_scalar('val/seg_acc', total_correct_seg / float(total_seen_seg), global_step=epoch)
            writer.add_scalar('val/m_acc', np.mean(correct_per_label / total_per_label), global_step=epoch)


            writer.add_scalar('val/loss', np.mean(all_loss), global_step=epoch)
            writer.add_scalar('val/loss_seg', np.mean(all_loss_seg), global_step=epoch)
            writer.add_scalar('val/loss_cls', np.mean(all_loss_cls), global_step=epoch)


            writer.add_scalar('train/loss_epoch', np.mean(loss_train_all), global_step=epoch)
            writer.add_scalar('train/loss_seg_epoch', np.mean(seg_loss_train_all), global_step=epoch)
            writer.add_scalar('train/loss_cls_epoch', np.mean(cls_loss_train_all), global_step=epoch)

            if total_correct / float(total_seen) > max_val_acc:
                max_val_acc = total_correct / float(total_seen)

                train_util_distributed.save_exp_parallel([generator, optimizer],
                                                         ['generator', 'g_opt'], exp_path=exp_dir,
                                                         epoch=0, epoch_name='best')

            if np.mean(correct_per_label / total_per_label) > max_val_macc:
                max_val_macc = np.mean(correct_per_label / total_per_label)

                train_util_distributed.save_exp_parallel([generator, optimizer],
                                                         ['generator', 'g_opt'], exp_path=exp_dir,
                                                         epoch=0, epoch_name='macc_best')

        if args.rank == 0:
            writer.flush()
if args.rank == 0:
    writer.close()
