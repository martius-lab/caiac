import glob
import os
import pipes
import sys

import numpy as np
import torch
from causal_slr.utils.general_utils import str2int


"Code in this file adapted from: Accelerating Reinforcement Learning with Learned Skill Priors: https://github.com/clvrai/spirl/tree/master"

class CheckpointHandler:
    @staticmethod
    def get_ckpt_name(epoch):
        return 'weights_ep{}.pth'.format(epoch)

    @staticmethod
    def get_epochs(path):
        checkpoint_names = glob.glob(os.path.abspath(path) + "/*.pth")
        if len(checkpoint_names) == 0:
            raise ValueError("No checkpoints found at {}!".format(path))
        processed_names = [file.split('/')[-1].replace('weights_ep', '').replace('.pth', '')
                           for file in checkpoint_names]
        epochs = list(filter(lambda x: x is not None, [
                      str2int(name) for name in processed_names]))
        return epochs

    @staticmethod
    def get_best_ckpt(path):
        checkpoint_names = glob.glob(os.path.abspath(path) + "/*.pth")
        if len(checkpoint_names) == 0:
            raise ValueError("No checkpoints found at {}!".format(path))

        potential_best = []
        for name in checkpoint_names:
            if 'best' in name:
                potential_best.append(name)
        if len(potential_best) == 0:
            raise ValueError("No BEST checkpoints found at {}!".format(path))
        if len(potential_best) > 1:
            raise ValueError(
                "Too many BEST checkpoints found at {}!".format(path))
        return potential_best[0]

    @staticmethod
    def get_last_ckpt(path):
        checkpoint_names = glob.glob(os.path.abspath(path) + "/*.pth")
        if len(checkpoint_names) == 0:
            raise ValueError("No checkpoints found at {}!".format(path))

        max_epoch = np.max(CheckpointHandler.get_epochs(path))
        last_file = CheckpointHandler.get_ckpt_name(max_epoch)
        return os.path.join(path, last_file)

    @staticmethod
    def get_resume_ckpt_file(resume, path):
        print("Loading from: {}".format(path))
        if resume == 'latest':
            max_epoch = np.max(CheckpointHandler.get_epochs(path))
            resume_file = CheckpointHandler.get_ckpt_name(max_epoch)
        elif str2int(resume) is not None:
            resume_file = CheckpointHandler.get_ckpt_name(resume)
        elif '.pth' not in resume:
            resume_file = resume + '.pth'
        else:
            resume_file = resume
        return os.path.join(path, resume_file)

    @staticmethod
    def load_weights(weights_file, model, load_step=False, load_opt=False, optimizer=None, strict=True):
        success = False
        if os.path.isfile(weights_file):
            print(("=> loading checkpoint '{}'".format(weights_file)))
            map_location = 'cpu' if model.device is None else model.device
            checkpoint = torch.load(weights_file, map_location=map_location)
            best_metric = checkpoint.get('best_metric', np.Inf)
            model.load_state_dict(checkpoint['state_dict'], strict=strict)
            if load_step:
                print('Loading step')
                start_epoch = checkpoint['epoch'] + 1
                global_step = checkpoint['global_step']
            if load_opt:
                print('Loading optimizer')
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                except (RuntimeError, ValueError) as e:
                    if not strict:
                        print(
                            "Could not load optimizer params because of changes in the network + non-strict loading")
                        pass
                    else:
                        raise e
            print(("=> loaded checkpoint '{}' (gradient step {})\n"
                  .format(weights_file, checkpoint['global_step'])))

            success = True
            if checkpoint.get('norm_params_input', False):
                print('Loading normalizer')
                model.norm_params_input = checkpoint['norm_params_input']
                model.norm_params_target = checkpoint['norm_params_target']

            else:
                print('No normalizer found. Passing.')
        else:
            raise ValueError(
                "Could not find checkpoint file in {}!".format(weights_file))

        if load_step:
            return global_step, start_epoch, success, best_metric
        else:
            return success

    @staticmethod
    def rename_parameters(dict, old, new):
        """ Renames parameters in the network by finding parameters that contain 'old' and replacing 'old' with 'new'
        """
        replacements = [key for key in dict if old in key]

        for key in replacements:
            dict[key.replace(old, new)] = dict.pop(key)


def save_git(base_dir):
    # save code revision
    print('Save git commit and diff to {}/git.txt'.format(base_dir))
    cmds = ["echo `git rev-parse HEAD` > {}".format(
        os.path.join(base_dir, 'git.txt')),
        "git diff >> {}".format(
        os.path.join(base_dir, 'git.txt'))]
    print(cmds)
    os.system("\n".join(cmds))


def save_cmd(base_dir):
    train_cmd = 'python ' + \
        ' '.join([sys.argv[0]] + [pipes.quote(s) for s in sys.argv[1:]])
    train_cmd += '\n\n'
    print('\n' + '*' * 80)
    print('Training command:\n' + train_cmd)
    print('*' * 80 + '\n')
    with open(os.path.join(base_dir, "cmd.txt"), "a") as f:
        f.write(train_cmd)


def load_by_key(checkpt_path, key, state_dict, device, epoch='latest'):
    """Loads weigths from checkpoint whose tag includes key."""
    checkpt_path = CheckpointHandler.get_resume_ckpt_file(epoch, checkpt_path)
    weights_dict = torch.load(checkpt_path, map_location=device)['state_dict']
    for checkpt_key in state_dict:
        if key in checkpt_key:
            print("Loading weights for {}".format(checkpt_key))
            state_dict[checkpt_key] = weights_dict[checkpt_key]
    return state_dict


def freeze_module(module):
    for p in module.parameters():
        if p.requires_grad:
            p.requires_grad = False


def freeze_modules(module_list):
    [freeze_module(module) for module in module_list]
