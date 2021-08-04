import torch
import torch.distributed as dist

from pathlib import Path

import pickle

import torch.nn.parallel


# taken from https://github.com/facebookresearch/maskrcnn-benchmark
def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def save_exp_parallel(objects, names, exp_path, epoch, epoch_name='epoch'):
    assert(len(objects) == len(names))

    for torch_object, name in zip(objects, names):
        with open('{}/{}_{}_{}.t7'.format(str(exp_path), name, epoch_name, epoch), 'wb') as f:
            if isinstance(torch_object, torch.nn.parallel.DistributedDataParallel):
                torch.save(torch_object.module.state_dict(), f)
            else:
                torch.save(torch_object.state_dict(), f)


def restore_exp(objects, names, device=torch.device('cuda:0'), verbose=True):
    assert(len(objects) == len(names))

    for torch_object, name in zip(objects, names):
        assert(Path(name).exists())

        if verbose:
            print('restoring form {}'.format(name))

        with open(name, 'rb') as f:
            torch_object.load_state_dict(torch.load(f, map_location=device))

    dist.barrier()
