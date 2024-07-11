import numpy as np
import torch
from causal_slr.utils.general_utils import map_recursive
from functools import partial
from torch.nn.modules import BatchNorm1d, BatchNorm2d, BatchNorm3d
from contextlib import contextmanager


def ten2ar(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    elif torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif np.isscalar(tensor):
        return tensor
    elif hasattr(tensor, 'to_numpy'):
        return tensor.to_numpy()
    else:
        import pdb
        pdb.set_trace()
        raise ValueError('input to ten2ar cannot be converted to numpy array')


def ar2ten(array, device, dtype=None):
    if isinstance(array, list) or isinstance(array, dict):
        return array

    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array).to(device)
    else:
        tensor = torch.tensor(array).to(device)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor


def map2torch(struct, device):
    print('mapping to device', device)
    """Recursively maps all elements in struct to torch tensors on the specified device."""
    return map_recursive(partial(ar2ten, device=device, dtype=torch.float32), struct)


def map2np(struct):
    """Recursively maps all elements in struct to numpy ndarrays."""
    return map_recursive(ten2ar, struct)


def switch_off_batchnorm_update(model):
    """Switches off batchnorm update in all submodules of model."""
    for module in model.modules():
        if isinstance(module, BatchNorm1d) \
                or isinstance(module, BatchNorm2d) \
                or isinstance(module, BatchNorm3d):
            module.eval()


def switch_on_batchnorm_update(model):
    """Switches on batchnorm update in all submodules of model."""
    for module in model.modules():
        if isinstance(module, BatchNorm1d) \
                or isinstance(module, BatchNorm2d) \
                or isinstance(module, BatchNorm3d):
            module.train()


@contextmanager
def no_batchnorm_update(model):
    """Switches off all batchnorm updates within context."""
    switch_off_batchnorm_update(model)
    yield
    switch_on_batchnorm_update(model)