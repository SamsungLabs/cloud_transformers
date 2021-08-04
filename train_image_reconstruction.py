import sys
import os
import numpy as np
import yaml
import time
import tqdm
import argparse

sys.path.append(os.path.realpath(__file__))

import torch

from datasets.image_point import ImageToPoint
from chamfer_extension import dist_chamfer
import emd_linear.emd_module as emd

from utils import train_util
from utils import train_util_distributed
from utils.pcd_utils import sphere_noise

import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed


torch.set_num_threads(1)


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="path to py model", required=True, default='.configs/config.yaml')

parser.add_argument("exp_name", help="name of the exp")
parser.add_argument("--master", required=True)
parser.add_argument("--rank", required=True, type=int)
parser.add_argument("--num_nodes", required=True, type=int)


args = parser.parse_args()
config_path = args.config


dist_backend = 'nccl'
# # Url used to setup distributed training
dist_url = "tcp://{}".format(args.master)
#
#
dist.init_process_group(backend=dist_backend,
                        init_method=dist_url,
                        rank=args.rank,
                        world_size=args.num_nodes)


with open(config_path, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

# check if we masked all other devices
assert(torch.cuda.device_count() == 1)
torch.cuda.set_device(0)


gen_model_file, gen_model_name = train_util.check_model_paths(cfg['model']['generator'])[0]


exp_descr = args.exp_name

# dataset part

completion_train = ImageToPoint(d_path=cfg['data']['path'],
                                split='train',
                                points=cfg['data']['gt_size'],
                                im_size=cfg['data']['im_size'])

completion_val = ImageToPoint(d_path=cfg['data']['path'],
                              split='val',
                              points=cfg['data']['gt_size'],
                              im_size=cfg['data']['im_size'])


train_sampler = torch.utils.data.distributed.DistributedSampler(completion_train)
val_sampler = torch.utils.data.distributed.DistributedSampler(completion_val)


dataloader_train = torch.utils.data.DataLoader(completion_train, batch_size=cfg['data']['batch_size'],
                                               shuffle=(train_sampler is None),
                                               num_workers=cfg['data']['num_workers'],
                                               pin_memory=True, sampler=train_sampler)
dataloader_val = torch.utils.data.DataLoader(completion_val, batch_size=cfg['data']['batch_size_val'],
                                             shuffle=False, num_workers=cfg['data']['num_workers'],
                                             pin_memory=True, sampler=val_sampler)

# model part
del cfg['model']['generator']
params_dict_gen = cfg['model']

np.random.seed(42 + args.rank)
torch.manual_seed(42 + args.rank)
torch.backends.cudnn.benchmark = True

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
    train_util_distributed.restore_exp(objects=[generator],
                                       names=[g_ckpt_path])

generator = torch.nn.parallel.DistributedDataParallel(torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator),
                                                      device_ids=[0],
                                                      output_device=0)

print('generator_params', sum(p.numel() for p in generator.parameters()))

if 'scale_lr' in  cfg['train']:
    opt_params_dict = [{'params': map(lambda p: p[1],
                                      filter(lambda p: not p[0].endswith('scale'),
                                             generator.named_parameters()))},
                       {'params': map(lambda p: p[1],
                                      filter(lambda p: p[0].endswith('scale'),
                                             generator.named_parameters())), 'lr': cfg['train']['scale_lr']}]
else:
    opt_params_dict = generator.parameters()

optimizer = train_util.make_optimizer(opt_params_dict, cfg['train']['optimizer'])

scheduler = train_util.make_scheduler(optimizer, cfg['train']['scheduler'])
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


max_val_loss = np.inf
EMD = emd.emdModule()


for epoch in range(cfg['train']['num_epochs']):
    train_sampler.set_epoch(epoch)
    generator.train()

    end = time.time()

    alpha = 1.0

    for i, (img, pcd_gt) in tqdm.tqdm(enumerate(dataloader_train)):
        data_time = time.time() - end

        img = img.cuda()
        pcd_gt = pcd_gt[:, :, None].cuda(non_blocking=True)

        noise = sphere_noise(pcd_gt.shape[0], pcd_gt.shape[-1], pcd_gt.device)

        reconstruction, lattices_sizes = generator(noise, img)

        dist, _ = EMD(reconstruction[:, :, 0].permute(0, 2, 1),
                      pcd_gt[:, :, 0].permute(0, 2, 1), 0.005, 50)
        loss = torch.sqrt(dist).mean(1).mean()

        with torch.no_grad():
            loss_chamfer = dist_chamfer.loss_chamfer_adj(reconstruction, pcd_gt)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_time = time.time() - end
        end = time.time()

        loss_all = train_util_distributed.reduce_loss_dict({'loss': loss,
                                                            'loss_chamfer': loss_chamfer})

        if args.rank == 0:
            writer.add_scalar('train/loss', loss_all['loss'].item(), global_step=data_iters)
            writer.add_scalar('train/loss_chamfer', loss_all['loss_chamfer'].item(), global_step=data_iters)

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

        if not scheduler_adaptive:
            scheduler.step(data_iters)

    # all_elements = []
    all_loss = []
    all_loss_chamfer = []

    fs, ps, rc = [], [], []

    generator.eval()
    val_sampler.set_epoch(epoch)

    with torch.no_grad():
        for i, (img, pcd_gt) in tqdm.tqdm(enumerate(dataloader_val)):
            img = img.cuda()
            pcd_gt = pcd_gt[:, :, None].cuda(non_blocking=True)

            noise = sphere_noise(pcd_gt.shape[0], pcd_gt.shape[-1], pcd_gt.device)

            reconstruction, _ = generator(noise, img)

            dist, _ = EMD(reconstruction[:, :, 0].permute(0, 2, 1),
                          pcd_gt[:, :, 0].permute(0, 2, 1),
                          0.004, 3000)

            loss = torch.sqrt(dist).mean(1).mean()

            loss_chamfer = dist_chamfer.loss_chamfer_adj(reconstruction, pcd_gt)

            loss_all = train_util_distributed.reduce_loss_dict({'loss': loss,
                                                                'loss_chamfer': loss_chamfer})
            all_loss.append(loss_all['loss'].item())
            all_loss_chamfer.append(loss_all['loss_chamfer'].item())

    if args.rank == 0 and epoch % cfg['train']['save_each_epoch'] == 0:
        train_util_distributed.save_exp_parallel([generator, optimizer],
                                                 ['generator', 'g_opt'],
                                                 exp_path=exp_dir, epoch=epoch,
                                                 epoch_name='epoch')

    if scheduler_adaptive:
        scheduler.step(np.mean(all_loss))

    if args.rank == 0:
        writer.add_scalar('val/val_loss_emd', np.mean(all_loss), global_step=epoch)
        writer.add_scalar('val/val_loss_chamfer', np.mean(all_loss_chamfer), global_step=epoch)

        if np.mean(all_loss) < max_val_loss:
            max_val_loss = np.mean(all_loss)

            train_util_distributed.save_exp_parallel([generator, optimizer],
                                                     ['generator', 'g_opt'], exp_path=exp_dir,
                                                     epoch=0, epoch_name='best')

    if args.rank == 0:
        writer.flush()
if args.rank == 0:
    writer.close()
