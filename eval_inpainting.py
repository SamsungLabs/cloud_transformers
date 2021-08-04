import sys
import os
sys.path.append(os.path.realpath(__file__))

import torch
from torch import nn


import argparse

from datasets.grnet_completion import ShapeNetDataLoader, DatasetSubset

from utils import train_util
from utils.pcd_utils import partial_postproces
import logging
from utils.grdnet_utils import AverageMeter
from utils.grdnet_utils import Metrics
from utils.grdnet_utils import ChamferDistance


import pickle
import torch.utils.data
import torch.utils.data.distributed

import yaml
import tqdm

import emd_linear.emd_module as emd

import numpy as np

def collate_fn(batch):
    taxonomy_ids = []
    model_ids = []
    data = {}

    for sample in batch:
        taxonomy_ids.append(sample[0])
        model_ids.append(sample[1])
        _data = sample[2]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)

    return taxonomy_ids, model_ids, data


torch.set_num_threads(1)


# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="path to py model", required=True, default='.configs/config.yaml')

parser.add_argument("exp_name", help="name of the exp")

args = parser.parse_args()
config_path = args.config



with open(config_path, 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

# check if we masked all other devices
assert(torch.cuda.device_count() == 1)
torch.cuda.set_device(0)


gen_model_file, gen_model_name = train_util.check_model_paths(cfg['model']['generator'])[0]


exp_descr = args.exp_name

del cfg['model']['generator']
params_dict_gen = cfg['model']

np.random.seed(42)
torch.manual_seed(42)

# torch.cuda.manual_seed_all(42)

# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# torch.cuda.manual_seed_all(42)

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


EMD = emd.emdModule()


save = cfg['train']['save']


def test_net(test_data_loader=None, test_writer=None):
    def var_or_cuda(x):
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)

        return x

    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader
        ds = ShapeNetDataLoader(complete_path=cfg['data']['gt_path'],
                                partial_path=cfg['data']['partial_path'],
                                category_file_path=cfg['data']['category_path'],
                                n_input=cfg['data']['input_size'],
                                n_output=cfg['data']['gt_size'],
                                n_renders=cfg['data']['n_renders']).get_dataset(DatasetSubset.TEST)

        test_data_loader = torch.utils.data.DataLoader(dataset=ds,
                                                       batch_size=1,
                                                       num_workers=1,
                                                       collate_fn=collate_fn,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Switch models to evaluation mode
    generator.eval()
    # Set up loss functions
    chamfer_dist = ChamferDistance()

    # Testing loop
    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['SparseLoss', 'DenseLoss'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    # Testing loop
    for model_idx, (taxonomy_id, model_id, data) in tqdm.tqdm(enumerate(test_data_loader)):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        model_id = model_id[0]

        with torch.no_grad():
            pcd_gt = 2 * data['gtcloud'].permute(0, 2, 1)[:, :, None].cuda(non_blocking=True)

            pcd_part_enc, pcd_part_noise = partial_postproces(2 * data['partial_cloud'], pcd_gt.shape[-1])

            pcd_part_enc = pcd_part_enc.permute(0, 2, 1)[:, :, None].cuda()
            pcd_part_noise = pcd_part_noise.permute(0, 2, 1).cuda()

            reconstruction, lattices_sizes = generator(pcd_part_noise, pcd_part_enc)
            dense_ptcloud = (reconstruction / 2)[:, :, 0].permute(0, 2, 1).contiguous()

            gt = data['gtcloud'].cuda()

            # sparse_loss = chamfer_dist(sparse_ptcloud, data['gtcloud'])
            dense_loss = chamfer_dist(dense_ptcloud, gt)
            test_losses.update([dense_loss.item() * 1000])
            _metrics = Metrics.get(dense_ptcloud, gt)

            current_data = (data['partial_cloud'].cpu(), data['gtcloud'].cpu(), dense_ptcloud.cpu(), _metrics)

            if save:
                with open('{}/{}_{}.pickle'.format(str(exp_dir), taxonomy_id, model_id), 'wb') as f:
                    pickle.dump(current_data, f)

            test_metrics.update(_metrics)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            logging.info('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                         (model_idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()
                                                                            ], ['%.4f' % m for m in _metrics]))

    # Print testing results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics:
        print(taxonomy_id, end='\t')
        print(category_metrics[taxonomy_id].count(0), end='\t')
        for value in category_metrics[taxonomy_id].avg():
            print('%.4f' % value, end='\t')
        print()

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    # Add testing results to TensorBoard

    return None

test_net()

