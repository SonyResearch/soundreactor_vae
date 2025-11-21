import os
import random

import numpy as np
import torch
from collections import defaultdict


def exists(x: torch.Tensor):
    return x is not None


def get_world_size():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    else:
        return torch.distributed.get_world_size()


def get_rank():
    """Get rank of current process."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    else:
        return torch.distributed.get_rank()


def print_once(*args):
    if get_rank() == 0:
        print(*args)


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters()) + \
        sum(p.numel() for p in model.buffers())


# def copy_state_dict(model, state_dict):
#     """Load state_dict to model, but only for keys that match exactly.

#     Args:
#         model (nn.Module): model to load state_dict.
#         state_dict (OrderedDict): state_dict to load.
#     """
#     model_state_dict = model.state_dict()
#     for key in state_dict:
#         if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape:
#             if isinstance(state_dict[key], torch.nn.Parameter):
#                 # backwards compatibility for serialized parameters
#                 state_dict[key] = state_dict[key].data
#             model_state_dict[key] = state_dict[key]

#     model.load_state_dict(model_state_dict, strict=False)
    


def copy_state_dict(model: torch.nn.Module,
                          ckpt_state: "dict[str, torch.Tensor]",
                          verbose: bool = True) -> None:
    model_state = model.state_dict()
    loaded_exact, loaded_fuzzy = [], []
    unused_ckpt, unmapped_model = [], []
    ckpt_used = set()

    for k, v in ckpt_state.items():
        if k in model_state and v.shape == model_state[k].shape:
            model_state[k] = v.data if isinstance(v, torch.nn.Parameter) else v
            loaded_exact.append(k)
            ckpt_used.add(k)

    shape2keys = defaultdict(list)
    for k, v in ckpt_state.items():
        if k not in ckpt_used:
            shape2keys[tuple(v.shape)].append(k)

    for mkey, mval in model_state.items():
        if mkey in loaded_exact:
            continue
        shape = tuple(mval.shape)
        if shape2keys[shape]:
            ckpt_key = shape2keys[shape].pop(0)
            ckpt_used.add(ckpt_key)
            cval = ckpt_state[ckpt_key]
            model_state[mkey] = cval.data if isinstance(cval, torch.nn.Parameter) else cval
            loaded_fuzzy.append((ckpt_key, mkey))

    unused_ckpt  = sorted(set(ckpt_state)  - ckpt_used)
    unmapped_model = sorted(set(model_state) - set(ckpt_state) - {m for _, m in loaded_fuzzy})

    model.load_state_dict(model_state, strict=False)
