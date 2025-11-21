import os
import random
from typing import Union

import torch
from torch import nn
import torch.distributed as dist
import numpy as np
from omegaconf import DictConfig


def exists(val):
    return val is not None


def print_once(*args, **kwargs):
    if (not dist.is_available()) or dist.get_rank() == 0:
        print(*args, **kwargs)

def sort_dict(D: dict):
    s_keys = sorted(D.keys())
    return {k: D[k] for k in s_keys}


def set_seeds(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(model: Union[nn.Module, nn.ModuleList, nn.ModuleDict, DictConfig]):
    # modules have to be set train mode
    def count_module(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad) + \
            sum(p.numel() for p in m.buffers())
    if isinstance(model, nn.Module):
        return count_module(model)
    elif isinstance(model, nn.ModuleList):
        return sum(count_module(m) for m in model)
    elif isinstance(model, (nn.ModuleList, DictConfig)):
        return sum(count_module(m) for m in model.values())
    else:
        raise RuntimeError(f'Invalid model type: {type(model)}')


def get_rank():
    return dist.get_rank() if dist.is_available() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_available() else 1


def set_param_grad(net, flg):
    for child in net.children():
        for param in child.parameters():
            param.requires_grad = flg


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    # https://github.com/LTH14/mar/blob/fe470ac24afbee924668d8c5c83e9fec60af3a73/util/misc.py#L291
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'diffloss' in name:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

#----------------------------------------------------------------------------
# Replace NaN/Inf with specified numerical values.

try:
    nan_to_num = torch.nan_to_num # 1.8.0a0
except AttributeError:
    def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None): # pylint: disable=redefined-builtin
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)

#----------------------------------------------------------------------------