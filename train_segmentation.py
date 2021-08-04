import sys
import os
import tqdm
import argparse
import random
import yaml
import time
from torch import nn
import numpy as np
from collections import defaultdict

from datasets.S3DIS_tools import iou_util_new
sys.path.append(os.path.realpath(__file__))

from datasets.s3dis_v2 import Indoor3DSemSeg
from utils import train_util
from utils import train_util_distributed
from utils.train_util import worker_init_fn

import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed


torch.set_num_threads(1)

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--config",
                    help="path to py model",
                    required=True, default='.configs/config_seg.yaml')
parser.add_argument("exp_name", help="name of the exp")
parser.add_argument("--master", required=True)
parser.add_argument("--rank", required=True, type=int)
parser.add_argument("--num_nodes", required=True, type=int)


args = parser.parse_args()
config_path = args.config


dist_backend = 'nccl'
# Url used to setup distributed training
dist_url = "tcp://{}".format(args.master)

assert(torch.cuda.device_count() == 1)
torch.cuda.set_device(0)


# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.manual_seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

dist.init_process_group(backend=dist_backend,
                        init_method=dist_url,
                        rank=args.rank,
                        world_size=args.num_nodes)


with open(config_path, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

# check if we masked all other devices

gen_model_file, gen_model_name = train_util.check_model_paths(cfg['model']['generator'])[0]
exp_descr = args.exp_name
aug = False
if 'aug' in cfg['data']:
    aug = cfg['data']['aug']

print('aug', aug)

indoor_seg_train = Indoor3DSemSeg(data_dir=cfg['data']['path'],
                                  train=True,
                                  num_points=cfg['data']['num_points'],
                                  aug=aug,
                                  test_area=cfg['data']['test_area'],
                                  data_precent=cfg['data']['data_percent'])

indoor_seg_val = Indoor3DSemSeg(data_dir=cfg['data']['path'],
                                train=False,
                                num_points=cfg['data']['num_points'],
                                test_area=cfg['data']['test_area'],
                                data_precent=cfg['data']['data_percent'])

train_sampler = torch.utils.data.distributed.DistributedSampler(indoor_seg_train)
val_sampler = torch.utils.data.distributed.DistributedSampler(indoor_seg_val)

dataloader_train = torch.utils.data.DataLoader(indoor_seg_train,
                                               batch_size=cfg['data']['batch_size'],
                                               shuffle=(train_sampler is None),
                                               num_workers=cfg['data']['num_workers'],
                                               pin_memory=False, sampler=train_sampler,
                                               worker_init_fn=worker_init_fn)

dataloader_val = torch.utils.data.DataLoader(indoor_seg_val,
                                             batch_size=cfg['data']['batch_size_val'],
                                             shuffle=False,
                                             num_workers=cfg['data']['num_workers'],
                                             pin_memory=False, sampler=val_sampler,
                                             worker_init_fn=worker_init_fn)

# model part
del cfg['model']['generator']
params_dict_gen = cfg['model']

if args.rank == 0:
    writer, exp_dir, full_desc = train_util.create_experiment(exp_descr,
                                                              params_dict={'exp_root': cfg['experiment']['root'],
                                                                           'writer_root': cfg['experiment']['writer_root'],
                                                                           'config_path': config_path})
    generator = train_util.get_model(gen_model_file, params_dict_gen,
                                     exp_dir=exp_dir).cuda()
else:
    generator = train_util.get_model(gen_model_file, params_dict_gen, exp_dir=None).cuda()

if 'restore' in cfg:
    print('restoring')
    g_ckpt_path = cfg['restore']['generator']
    train_util.restore_exp_fix(objects=[generator],
                               names=[g_ckpt_path])


generator = torch.nn.parallel.DistributedDataParallel(torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator),
                                                      device_ids=[0],
                                                      output_device=0)

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
scheduler_adaptive = False

if 'restore' in cfg:
    print('restoring')
    g_opt_ckpt_path = cfg['restore']['optimizer']
    train_util_distributed.restore_exp(objects=[optimizer],
                                       names=[g_opt_ckpt_path])

    if 'new_lr' in cfg['restore']:
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg['restore']['new_lr']

show_each = cfg['train']['show_each']
show_iters = 0
data_iters = 0

scheduler_base = train_util.make_scheduler(optimizer, cfg['train']['scheduler'])
scheduler = scheduler_base


cross_entropy_loss = nn.CrossEntropyLoss()


for epoch in range(cfg['train']['num_epochs']):
    train_sampler.set_epoch(epoch)
    generator.train()
    all_loss_train = []

    end = time.time()

    confusion_train = iou_util_new.ConfusionMatrix(number_of_labels=13)

    for i, (pcd, labels) in tqdm.tqdm(enumerate(dataloader_train)):
        data_time = time.time() - end

        pcd = pcd.permute(0, 2, 1)[:, :, None].cuda()
        labels = labels.cuda(non_blocking=True)

        pred, lattices_sizes = generator(pcd)

        loss = cross_entropy_loss(pred[:, :, 0], labels)
        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        batch_time = time.time() - end
        end = time.time()

        loss_all = train_util_distributed.reduce_loss_dict({'loss': loss})
        all_loss_train.append(loss_all['loss'].item())

        pred_this = np.argmax(pred[:, :, 0].detach().cpu().numpy(), axis=1)
        labels_this = labels.detach().cpu().numpy()

        all_gathered = train_util_distributed.all_gather((pred_this, labels_this))

        if args.rank == 0:
            for pred_np, labels_np in all_gathered:
                confusion_train.count_predicted_batch_hard(labels_np.flatten(), pred_np.flatten())

        if args.rank == 0:
            writer.add_scalar('train/loss', loss_all['loss'].item(), global_step=data_iters)
            writer.add_scalar('train/data_time', data_time, global_step=data_iters)
            writer.add_scalar('train/batch_time', batch_time, global_step=data_iters)

            # lattice stats
            for show_id, value in enumerate(lattices_sizes):
                writer.add_scalar('train/lattice_{}'.format(show_id),
                                  value[0], global_step=data_iters)
                writer.add_scalar('train/norm_l_feat_{}'.format(show_id),
                                  value[1].item(), global_step=data_iters)
                writer.add_scalar('train/norm_l_feat_var_{}'.format(show_id),
                                  value[2].item(), global_step=data_iters)

            if data_iters % cfg['train']['save_each'] == 0 and data_iters > 0:
                generator.eval()

                train_util_distributed.save_exp_parallel([generator, optimizer],
                                                         ['generator', 'g_opt'], exp_path=exp_dir,
                                                         epoch=data_iters, epoch_name='iter')
                generator.train()

        data_iters += 1

        # if not scheduler_adaptive:
        scheduler.step(data_iters)

        del pcd, labels

    if args.rank == 0:
        print('on train: ', iou_util_new.return_metrics_dict(confusion_train))

    if args.rank == 0 and epoch % cfg['train']['save_each_epoch'] == 0 and epoch > 0:
        print('saving ckpt')
        train_util.save_exp([generator, optimizer],
                            ['generator', 'g_opt'], exp_path=exp_dir, epoch=epoch, epoch_name='epoch')

    if epoch % cfg['train']['val_step'] == 0:
        all_loss = []

        confusion_val = iou_util_new.ConfusionMatrix(number_of_labels=13)

        generator.eval()
        val_sampler.set_epoch(epoch)

        lattice_info = defaultdict(list)

        with torch.no_grad():
            for i, (pcd, labels) in tqdm.tqdm(enumerate(dataloader_val)):
                pcd = pcd.permute(0, 2, 1)[:, :, None].cuda()
                labels = labels.cuda(non_blocking=True)

                pred, lattices_sizes = generator(pcd)

                for ind, value in enumerate(lattices_sizes):
                    lattice_info['lattice_{}'.format(ind)].append(value[0])
                    lattice_info['norm_l_feat_{}'.format(ind)].append(value[1].item())
                    lattice_info['norm_l_feat_var_{}'.format(ind)].append(value[2].item())

                loss = cross_entropy_loss(pred[:, :, 0], labels)

                loss_all = train_util_distributed.reduce_loss_dict({'loss': loss})
                all_loss.append(loss_all['loss'].item())

                pred_this = np.argmax(pred[:, :, 0].detach().cpu().numpy(), axis=1)
                labels_this = labels.detach().cpu().numpy()

                all_gathered = train_util_distributed.all_gather((pred_this, labels_this))

                if args.rank == 0:
                    for pred_np, labels_np in all_gathered:
                        confusion_val.count_predicted_batch_hard(labels_np.flatten(), pred_np.flatten())

                del pcd, labels

        if args.rank == 0:
            metrics_dict = iou_util_new.return_metrics_dict(confusion_val)

            for metric_name, metric_value in metrics_dict.items():
                writer.add_scalar('val/' + metric_name, metric_value, global_step=epoch)

            writer.add_scalar('val/loss', np.mean(all_loss), global_step=epoch)
            writer.add_scalar('train/loss_epoch', np.mean(all_loss_train), global_step=epoch)

        if args.rank == 0:
            writer.flush()

if args.rank == 0:
    writer.close()
