import numpy as np
import torch

from contextlib import contextmanager

import time
import random

from copy import deepcopy
from functools import partial, reduce
import collections
from smart_settings.param_classes import ImmutableAttributeDict
import yaml
import os
from torch.optim import Adam, RMSprop, SGD
from collections import defaultdict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, digits=None):
        """
        :param digits: number of digits returned for average value
        """
        self._digits = digits
        self.reset()

    def reset(self):
        self.val = 0
        self._avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self._avg = self.sum / self.count

    @property
    def avg(self):
        if self._digits is not None:
            return np.round(self._avg, self._digits)
        else:
            return self._avg


class AverageTimer(AverageMeter):
    """Times whatever is inside the with self.time(): ... block, exposes average etc like AverageMeter."""
    @contextmanager
    def time(self):
        self.start = time.time()
        yield
        self.update(time.time() - self.start)


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, d):
        self = d


def recursive_objectify(nested_dict, make_immutable=True):
    "Turns a nested_dict into a nested AttributeDict"
    result = deepcopy(nested_dict)
    for k, v in result.items():
        if isinstance(v, collections.Mapping):
            result = dict(result)
            result[k] = recursive_objectify(v, make_immutable)

    returned_result = AttrDict(result)
    return returned_result


def str2int(str):
    try:
        return int(str)
    except ValueError:
        return None


@contextmanager
def dummy_context():
    yield


def get_clipped_optimizer(*args, optimizer_type=None, **kwargs):
    assert optimizer_type is not None  # need to set optimizer type!

    class ClipGradOptimizer(optimizer_type):
        def __init__(self, *args, gradient_clip=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.gradient_clip = gradient_clip

        def step(self, *args, **kwargs):
            # This doesnt work conversion to numpy faisl. Lets do it in main train method
            # if self.gradient_clip is not None:
            #     params = np.concatenate([group['params'] for group in self.param_groups])
            #     torch.nn.utils.clip_grad_norm_(params, self.gradient_clip)

            super().step(*args, **kwargs)

    return ClipGradOptimizer(*args, **kwargs)


@contextmanager
def timing(text, name=None, interval=10):
    start = time.time()
    yield
    elapsed = time.time() - start

    if name:
        if not hasattr(timing, name):
            setattr(timing, name, AverageMeter())
        meter = getattr(timing, name)
        meter.update(elapsed)
        if meter.count % interval == 0:
            print("{} {}".format(text, meter.avg))
        return

    print("{} {}".format(text, elapsed))


class RecursiveAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0

    def update(self, val):
        self.val = val
        if self.sum is None:
            self.sum = val
        else:
            self.sum = map_recursive_list(lambda x, y: x + y, [self.sum, val])
        self.count += 1
        self.avg = map_recursive(lambda x: x / self.count, self.sum)


def make_recursive(fn, *argv, **kwargs):
    """ Takes a fn and returns a function that can apply fn on tensor structure
     which can be a single tensor, tuple or a list. """

    def recursive_map(tensors):
        if tensors is None:
            return tensors
        elif isinstance(tensors, list) or isinstance(tensors, tuple):
            return type(tensors)(map(recursive_map, tensors))
        elif isinstance(tensors, dict):
            return type(tensors)(map_dict(recursive_map, tensors))
        elif isinstance(tensors, torch.Tensor) or isinstance(tensors, np.ndarray):
            return fn(tensors, *argv, **kwargs)
        else:
            try:
                return fn(tensors, *argv, **kwargs)
            except Exception as e:
                print(
                    "The following error was raised when recursively applying a function:")
                print(e)
                raise ValueError(
                    "Type {} not supported for recursive map".format(type(tensors)))

    return recursive_map


def make_recursive_list(fn):
    """ Takes a fn and returns a function that can apply fn across tuples of tensor structures,
     each of which can be a single tensor, tuple or a list. """

    def recursive_map(tensors):
        if tensors is None:
            return tensors
        elif isinstance(tensors[0], list) or isinstance(tensors[0], tuple):
            return type(tensors[0])(map(recursive_map, zip(*tensors)))
        elif isinstance(tensors[0], dict):
            return map_dict(recursive_map, listdict2dictlist(tensors))
        elif isinstance(tensors[0], torch.Tensor):
            return fn(*tensors)
        else:
            try:
                return fn(*tensors)
            except Exception as e:
                print(
                    "The following error was raised when recursively applying a function:")
                print(e)
                raise ValueError(
                    "Type {} not supported for recursive map".format(type(tensors)))

    return recursive_map


recursively = make_recursive


def map_recursive(fn, tensors):
    return make_recursive(fn)(tensors)


def map_recursive_list(fn, tensors):
    return make_recursive_list(fn)(tensors)


def map_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    return type(d)(map(lambda kv: (kv[0], fn(kv[1])), d.items()))


def rec_map_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    # return type(d)({k: (rec_map_dict(fn, v) if isinstance(v, dict) else fn(v)) for k, v in d.items()})
    return type(d)(map(lambda kv: (kv[0], rec_map_dict(fn, kv[1]) if isinstance(kv[1], dict) else fn(kv[1])), d.items()))


def listdict2dictlist(LD):
    """ Converts a list of dicts to a dict of lists """
    # Take intersection of keys
    keys = reduce(lambda x, y: x & y, (map(lambda d: d.keys(), LD)))
    return AttrDict({k: [dic[k] for dic in LD] for k in keys})


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def prefix_dict(d, prefix):
    """Adds the prefix to all keys of dict d."""
    return type(d)({prefix+k: v for k, v in d.items()})


class RolloutStorage:
    """Can hold multiple rollouts, can compute statistics over these rollouts."""

    def __init__(self):
        self.rollouts = []

    def append(self, rollout):
        """Adds rollout to storage."""
        self.rollouts.append(rollout)

    def rollout_stats(self):
        """Returns AttrDict of average statistics over the rollouts."""
        assert self.rollouts    # rollout storage should not be empty
        reward, success_rate, episode_len, count, count_task, success_rate_task = 0, 0, 0, 0, defaultdict(
            int), defaultdict(int)
        for rollout in self.rollouts:
            reward += np.sum(rollout.reward)
            success_rate += rollout.ep_success
            episode_len += rollout.ep_len

            success_rate_task[rollout.task] += rollout.ep_success
            count_task[rollout.task] += 1
            count += 1

        stats = AttrDict(avg_reward=reward/count,
                         success_rate=success_rate/count,
                         avg_episode_len=episode_len/count)

        for k, v in success_rate_task.items():
            stats['success_rate_'+str(k)] = v/count_task[k]

        return stats

    def reset(self):
        del self.rollouts
        self.rollouts = []

    def get(self):
        return self.rollouts

    def __contains__(self, key):
        return self.rollouts and key in self.rollouts[0]


def np2obj(np_array):
    if isinstance(np_array, list) or np_array.size > 1:
        return [e[0] for e in np_array]
    else:
        return np_array[0]


def set_seeds(seed=0, cuda_deterministic=True):
    """Sets all seeds and disables non-determinism in cuDNN backend."""
    print(f'Setting seed={seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def save_config(conf, exp_conf_path):
    # Convert AttrDict conf to Dict so that yaml can save it
    def tostrdict(d):
        new_str_dict = {}
        for k, v in d.items():
            if issubclass(type(v), (dict, ImmutableAttributeDict)):
                v = tostrdict(v)
            else:
                if type(v) not in [bool, float, int, str, list] and v is not None:
                    try:
                        v = str(v).split('<class \'')[1][:-2]
                    except:
                        raise ValueError(
                            f'cannot convert {type(v)} of {v} into yaml compatatible format')
            new_str_dict[k] = v

        return new_str_dict
    dict_conf = tostrdict(conf)

    with open(exp_conf_path, 'w') as outfile:
        print('Saving config in ', exp_conf_path)
        yaml.dump(dict_conf, outfile,
                  default_flow_style=False, sort_keys=False)


def get_exp_dir():
    # Used by rl/train.py
    return os.environ['EXP_DIR']


def get_data_dir():
    return os.environ['DATA_DIR']


def get_optimizer_class(params):
    # TODO: clean this since we use raw simple Adam optimizer :S
    optim = params.optimizer
    if optim == 'adam':
        get_optim = partial(get_clipped_optimizer, optimizer_type=Adam, betas=(
            params.adam_beta, 0.999))
    elif optim == 'radam':
        get_optim = partial(get_clipped_optimizer, optimizer_type=RAdam, betas=(
            params.adam_beta, 0.999))
    elif optim == 'rmsprop':
        get_optim = partial(
            get_clipped_optimizer, optimizer_type=RMSprop, momentum=params.momentum)
    elif optim == 'sgd':
        get_optim = partial(get_clipped_optimizer,
                            optimizer_type=SGD, momentum=params.momentum)
    else:
        raise ValueError("Optimizer '{}' not supported!".format(optim))
    return partial(get_optim, gradient_clip=params.gradient_clip)
