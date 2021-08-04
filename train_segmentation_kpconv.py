import sys
import os
import yaml
import time
import numpy as np
sys.path.append(os.path.realpath(__file__))

import argparse
import random

from utils import train_util
from utils import train_util_distributed

import torch.distributed as dist

import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms

from datasets.s3dis_closer_logger import setup_logger


from datasets.s3dis_closer_utils import PointcloudToTensor, PointcloudRandomRotate, PointcloudScaleAndJitter
from datasets.s3dis_closer import S3DISSeg
from datasets.s3dis_closer_train import train, validate, MaskedCrossEntropy


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

# dataset part

aug = False
if 'aug' in cfg['data']:
    aug = cfg['data']['aug']

print('aug', aug)


class FakeCFG:
    def __init__(self):
        self.data_root = cfg['data']['path']
        self.batch_size = cfg['data']['batch_size']
        self.num_points = cfg['data']['num_points']
        self.epochs = cfg['train']['num_epochs']
        self.num_workers = cfg['data']['num_workers']

        self.num_steps = 2000
        self.input_features_dim = 4

        self.num_classes = 13

        self.sampleDl = 0.04

        self.in_radius = 2.0

        self.print_freq = 10

        self.x_angle_range = 0.0
        self.y_angle_range = 0.0
        self.z_angle_range = 3.1415926

        self.scale_low = 0.7
        self.scale_high = 1.3

        self.noise_std = 0.001
        self.noise_clip = 0.05
        self.translate_range = 0.0

        self.color_drop = 0.2

        self.augment_symmetries = [1, 0, 0]



config = FakeCFG()

train_transforms = transforms.Compose([
    PointcloudToTensor(),
    PointcloudRandomRotate(x_range=config.x_angle_range, y_range=config.y_angle_range,
                                   z_range=config.z_angle_range),
    PointcloudScaleAndJitter(scale_low=config.scale_low, scale_high=config.scale_high,
                                     std=config.noise_std, clip=config.noise_clip,
                                     augment_symmetries=config.augment_symmetries),
])

test_transforms = transforms.Compose([
    PointcloudToTensor(),
])

train_dataset = S3DISSeg(input_features_dim=config.input_features_dim,
                         subsampling_parameter=config.sampleDl, color_drop=config.color_drop,
                         in_radius=config.in_radius, num_points=config.num_points,
                         num_steps=config.num_steps, num_epochs=config.epochs,
                         data_root=config.data_root, transforms=train_transforms,
                         split='train')
val_dataset = S3DISSeg(input_features_dim=config.input_features_dim,
                       subsampling_parameter=config.sampleDl, color_drop=config.color_drop,
                       in_radius=config.in_radius, num_points=config.num_points,
                       num_steps=config.num_steps, num_epochs=20,
                       data_root=config.data_root, transforms=test_transforms,
                       split='val')

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=False)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=config.batch_size,
                                           shuffle=False,
                                           num_workers=config.num_workers,
                                           pin_memory=True,
                                           sampler=train_sampler,
                                           drop_last=True)

val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=config.batch_size,
                                         shuffle=False,
                                         num_workers=config.num_workers,
                                         pin_memory=True,
                                         sampler=val_sampler,
                                         drop_last=False)
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

runing_vote_logits = [np.zeros((config.num_classes, l.shape[0]), dtype=np.float32) for l in
                      val_loader.dataset.sub_clouds_points_labels]

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


label_smooth = False

if 'label_smooth' in cfg['train']:
    label_smooth = cfg['train']['label_smooth']


criterion = MaskedCrossEntropy()
model = generator

logger = setup_logger(output=None, distributed_rank=dist.get_rank(), name="s3dis")

for epoch in range(1, cfg['train']['num_epochs'] + 1):
    train_sampler.set_epoch(epoch)
    generator.train()
    all_loss_train = []

    train_loader.sampler.set_epoch(epoch)
    val_loader.sampler.set_epoch(epoch)
    train_loader.dataset.epoch = epoch - 1
    tic = time.time()
    loss = train(epoch, train_loader, model, criterion, optimizer, scheduler, config, logger=logger)

    logger.info('epoch {}, total time {:.2f}, lr {:.5f}'.format(epoch,
                                                                (time.time() - tic),
                                                                optimizer.param_groups[0]['lr']))
    if epoch % cfg['train']['val_step'] == 0:
        validate(epoch, val_loader, model, criterion, runing_vote_logits, config, logger=logger, num_votes=2)

    if args.rank == 0 and epoch % cfg['train']['save_each_epoch'] == 0 and epoch > 0:
        print('saving ckpt')
        train_util.save_exp([generator, optimizer],
                            ['generator', 'g_opt'], exp_path=exp_dir, epoch=epoch, epoch_name='epoch')

    if args.rank == 0 and writer is not None:
        # tensorboard logger
        writer.add_scalar('ins_loss', loss, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

validate('Last', val_loader, model, criterion, runing_vote_logits, config, logger=logger, num_votes=20)
