from causal_slr.models.skill_mdl import SkillMdl
from torch import nn
from causal_slr.dataloaders.kitchen.dataloader import D4RLSequenceSplitDataset, FactorizedForwardDataset
import gym
import d4rl
from causal_slr.utils.losses import *

from causal_slr.envs.kitchen_env import MyKitchenEnv

from causal_slr import modules


from causal_slr.modules.layers import *

import torch


def get_model(class_name, **hyper_params):
    model_dict = {
        'FactorizedMLP': modules.FactorizedMLP,
        'Transformer': modules.Transformer}

    if hyper_params is not None:
        return model_dict[class_name](**hyper_params)
    else:
        return model_dict[class_name]


def get_skill_model(class_name, hyper_params=None):
    model_dict = {
        'SkillMdl': SkillMdl,
        'ClSPiRLMdl': SkillMdl}

    if hyper_params is not None:
        return model_dict[class_name](hyper_params)
    else:
        return model_dict[class_name]


def get_layer(class_name, hyper_params=None):
    layer_dict = {'GaussianLikelihoodHead': GaussianLikelihoodHead,
                  'linear': nn.Linear}
    if hyper_params is not None:
        return layer_dict[class_name](**hyper_params)
    else:
        return layer_dict[class_name]


def get_init_weights(class_name):
    init_dict = {'orthogonal_': torch.nn.init.orthogonal_,
                 'zeros_': torch.nn.init.zeros_,
                 'xavier_normal_': torch.nn.init.xavier_normal_,
                 'xavier_uniform_': torch.nn.init.xavier_uniform_}
    return init_dict[class_name]


def get_dataloader(class_name, hyper_params=None):
    layer_dict = {  # 'SkillD4RLSequenceSplitDataset': SkillD4RLSequenceSplitDataset,
        'D4RLSequenceSplitDataset': D4RLSequenceSplitDataset,
        'FactorizedForwardDataset': FactorizedForwardDataset
    }
    if hyper_params is not None:
        return layer_dict[class_name](**hyper_params)
    else:
        return layer_dict[class_name]


def get_env(class_name, hyper_params=None):
    if 'kitchen' in class_name:
        return MyKitchenEnv(gym.make(class_name), **hyper_params)

    return gym.make(class_name)


def get_loss(class_name, hyper_params=None):
    losses = {'nll_loss': gaussian_nll_loss,
              'beta_nll_loss': beta_nll_loss,
              'mse_loss': nn.MSELoss()
              }

    if hyper_params is not None:
        return losses[class_name](hyper_params)
    else:
        return losses[class_name]
