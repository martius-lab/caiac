import os
import csv
import numpy as np
from causal_slr.utils.wandb import WandBLogger
import yaml

LOG_WANDB = True
WANDB_PROJECT_NAME = 'causal_rl'
WANDB_ENTITY_NAME = 'nuria'
WANDB_MODE = "default"  # "disabled"


def discounted_return(rewards, gamma, reward_offset=True):
    rewards += (1 if reward_offset else 0)
    discount_weights = np.power(gamma, np.arange(
        rewards.shape[1])).reshape(1, rewards.shape[1])
    dis_return = (rewards * discount_weights).sum(axis=1)
    undis_return = rewards.sum(axis=1)
    return dis_return, undis_return


def update_avg_dict(avg_dict, new_dict):
    for k, v in new_dict.items():
        avg_dict[k] = (avg_dict[k] if k in avg_dict else []) + [v]


class Logger:

    def __init__(self, args):
        global LOG_WANDB
        LOG_WANDB = False if args.dont_save else LOG_WANDB
        self.args = args
        file_path = os.path.join(self.args.working_dir, 'metrics.csv')
        self.keys = True if os.path.isfile(file_path) else None
        self.file = open(file_path, 'a', newline='')
        self.csv_writer = csv.writer(self.file)
        if LOG_WANDB:
            self.wandb_logger = self.setup_logging(args)
        self.step = 0

    def log(self, d):
        if not self.keys:
            self.keys = list(d.keys())
            self.csv_writer.writerow(self.keys)
        self.csv_writer.writerow(list(d.values()))
        self.file.flush()
        if LOG_WANDB:
            self.wandb_logger.log_scalar_dict(d)
        self.step += 1

    def log_hparams(self, d):
        self.log(d)

    def close(self):
        self.file.close()

    def setup_logging(self, conf):
        if not os.path.exists(os.path.join(self.args.working_dir, 'id_wandb')):
            id_wandb = None
        else:
            with open(os.path.join(self.args.working_dir, 'id_wandb'), 'r') as file:
                id_wandb = yaml.safe_load(file)
        sweep_name = self.args.working_dir
        group = self.args.env + '_' + self.args.scorer_cls + \
            '_' + sweep_name
        logger = WandBLogger(self.args.working_dir, WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,
                             path=self.args.working_dir, conf=conf, wandb_mode=WANDB_MODE, id=id_wandb, group=group)

        return logger


class Normalizer:

    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        self.sum = np.zeros(self.size, np.float32)
        self.sumsq = np.zeros(self.size, np.float32)
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
        self.count = 0

    def update(self, v):
        v = v.reshape(-1, self.size)
        self.sum += v.sum(axis=0)
        self.sumsq += (np.square(v)).sum(axis=0)
        self.count += v.shape[0]

    def recompute_stats(self):
        self.mean = self.sum / self.count
        self.std = np.sqrt(np.maximum(np.square(
            self.eps), (self.sumsq / self.count) - np.square(self.sum / self.count)))

    def normalize(self, v, clip_range=None):
        clip_range = clip_range or self.default_clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)
