import shutil
from pathlib import Path

import datetime

# todo migrate
from tensorboardX import SummaryWriter

import torch

from collections import OrderedDict

import numpy as np

import random


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    random.seed(np.random.get_state()[1][0] + worker_id)


def get_model(model_file, params_dict, exp_dir=None):
    dict_env = dict()
    with open(str(model_file), 'r') as f:
        exec(f.read(), dict_env)
    model = dict_env['Model'](**params_dict)

    if exp_dir is not None:
        # copy model py
        assert(exp_dir.exists())
        shutil.copy2(str(model_file), str(exp_dir))

    return model


def check_model_paths(*paths):
    result = []

    for model_path in paths:
        model_file = Path(model_path)

        assert (model_file.exists())
        assert (model_file.suffix == '.py')

        model_name = model_file.name[:-3]

        result.append((model_path, model_name))

    return result


def create_experiment(*desc, params_dict):
    exp_path = Path(params_dict['exp_root'])
    writer_path = Path(params_dict['writer_root'])

    assert(writer_path.exists())

    date = datetime.datetime.now().strftime('%d_%m_%y_%H_%M_%S')
    full_desc = '_'.join([*desc, date])

    writer = SummaryWriter(str(writer_path.joinpath(full_desc)))
    exp_dir = exp_path.joinpath(full_desc)
    exp_dir.mkdir(parents=True)

    if 'config_path' in params_dict:
        config_path = Path(params_dict['config_path'])
        assert(config_path.exists())
        shutil.copy2(str(config_path), str(exp_dir))

    return writer, exp_dir, full_desc


def save_exp(objects, names, exp_path, epoch, epoch_name='epoch'):
    assert(len(objects) == len(names))

    for torch_object, name in zip(objects, names):
        with open('{}/{}_{}_{}.t7'.format(str(exp_path), name, epoch_name, epoch), 'wb') as f:
            torch.save(torch_object.state_dict(), f)


def restore_exp(objects, names, device, verbose=True, strict=True):
    assert(len(objects) == len(names))

    for torch_object, name in zip(objects, names):
        assert(Path(name).exists())

        if verbose:
            print('restoring form {}'.format(name))

        with open(name, 'rb') as f:
            if strict:
                torch_object.load_state_dict(torch.load(f, map_location=device), strict=True)
            else:
                torch_object.load_state_dict(torch.load(f, map_location=device))


def restore_exp_fix(objects, names, device=torch.device('cuda:0'), verbose=True):
    assert(len(objects) == len(names))

    for torch_object, name in zip(objects, names):
        assert(Path(name).exists())

        if verbose:
            print('restoring form {}'.format(name))

        with open(name, 'rb') as f:
            statedict = torch.load(f, map_location=device)
            new_state_dict = OrderedDict()

            for key, value in statedict.items():
                if key.startswith('module.'):
                    new_key = key[7:]
                else:
                    new_key = key
                new_state_dict[new_key] = value

            torch_object.load_state_dict(new_state_dict, strict=True)


def make_optimizer(params_dict, opt_cfg):
    optimizer = getattr(torch.optim, opt_cfg['type'])

    del opt_cfg['type']

    return optimizer(params_dict, **opt_cfg)


def make_scheduler(optimizer, scheduler_cfg):
    scheduler = getattr(torch.optim.lr_scheduler, scheduler_cfg['type'])
    del scheduler_cfg['type']

    return scheduler(optimizer, **scheduler_cfg)
