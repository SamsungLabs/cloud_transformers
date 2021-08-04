import sys
import os
sys.path.append(os.path.realpath(__file__))

import torch
import yaml
import pickle
import numpy as np
import tqdm
import argparse
from collections import defaultdict
from datasets.image_point import ImageToPoint
import emd_linear.emd_module as emd

from utils import train_util
from utils.pcd_utils import sphere_noise

from utils.f1_metric import get_f1_scores, get_f1_scores_merge
import torch.utils.data
import torch.utils.data.distributed


torch.set_num_threads(1)


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


completion_val = ImageToPoint(d_path=cfg['data']['path'],
                              split='test',
                              points=10000,
                              im_size=cfg['data']['im_size'])


dataloader_val = torch.utils.data.DataLoader(completion_val, batch_size=cfg['data']['batch_size'],
                                             shuffle=False, num_workers=cfg['data']['num_workers'],
                                             pin_memory=True)

# model part
del cfg['model']['generator']
params_dict_gen = cfg['model']

np.random.seed(42)
torch.manual_seed(42)

generator = train_util.get_model(gen_model_file, params_dict_gen, exp_dir=None).cuda()

if 'restore' in cfg:
    print('restoring')
    g_ckpt_path = cfg['restore']['generator']
    train_util.restore_exp_fix(objects=[generator],
                               names=[g_ckpt_path])

print('generator_params', sum(p.numel() for p in generator.parameters()))

max_val_loss = np.inf
EMD = emd.emdModule()

generator.eval()


fs = defaultdict(list)
ps = defaultdict(list)
rc = defaultdict(list)

writer, exp_dir, full_desc = train_util.create_experiment(exp_descr,
                                                          params_dict={'exp_root': cfg['experiment']['root'],
                                                                       'writer_root': cfg['experiment']['writer_root'],
                                                                       'config_path': config_path})

all_data = []
save = True

with torch.no_grad():
    for i, (img, pcd_gt, data_info) in tqdm.tqdm(enumerate(dataloader_val)):
        img = img.cuda()
        pcd_gt = pcd_gt[:, :, None].cuda(non_blocking=True)
        noise = sphere_noise(pcd_gt.shape[0], 4096 * 2, pcd_gt.device)
        noise_v2 = sphere_noise(pcd_gt.shape[0], 4096 * 2, pcd_gt.device)

        reconstruction, _ = generator(noise, img)
        reconstruction_v2, _ = generator(noise_v2, img)

        scores = get_f1_scores_merge(pcd=reconstruction[:, :, 0], pcd_2=reconstruction_v2[:, :, 0], pcd_gt=pcd_gt[:, :, 0], th=0.01)

        current_data = (noise.detach().cpu().numpy(),
                        noise_v2.detach().cpu().numpy(),
                        reconstruction.detach().cpu().numpy(),
                        reconstruction_v2.detach().cpu().numpy(),
                        img.detach().cpu().numpy(),
                        pcd_gt.detach().cpu().numpy(),
                        scores)

        for score, model_id in zip(scores, data_info):
            fs[model_id].append(score[0])
            ps[model_id].append(score[1])
            rc[model_id].append(score[2])

        if save:
            with open('{}/{}_{}.pickle'.format(str(exp_dir), 'data', str(i)), 'wb') as f:
                pickle.dump(current_data, f)

print('done, printing results')

for key, value in fs.items():
    print(key, np.mean(value))
