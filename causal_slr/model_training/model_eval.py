import torch
from collections import defaultdict

from causal_slr.utils.general_utils import RecursiveAverageMeter, map_dict, rec_map_dict
import numpy as np
from causal_slr.utils.general_utils import AttrDict, recursive_objectify

import gym
from causal_slr.utils import conf_utils
from matplotlib import pyplot as plt
import glob
import yaml
from causal_slr.dataloaders.kitchen.dataloader import get_data


class WorldModelEvaluator:

    def __init__(self, conf, model, train_loader, mdp_path=None):
        filelist = glob.glob(f'{mdp_path}/*.yaml')
        if len(filelist) == 1:
            config_path = filelist[0]
        else:
            raise FileNotFoundError(
                f'Couldnt find world model file in {filelist}')
        with open(config_path, 'r') as file:
            conf = yaml.safe_load(file)
        conf = recursive_objectify(AttrDict(conf))
        path = conf.path_data if len(conf.path_data)>0 else conf.env_config.name
        try:
            data_percentages = {'expert_percent': conf.data_params.expert_percent, 'random_percent': conf.data_params.random_percent, 'shrink_dataset': conf.data_params.shrink_dataset}
        except:
            data_percentages = {}

        self.data = get_data(path, **data_percentages)
        self.train_loader = train_loader
        self.model_test = model
        self.conf = conf
        self.env = conf_utils.get_env(
            conf.env_config.name, conf.env_config)
        self.input_factorizer = self.env.get_input_factorizer
        self.output_factorizer = self.env.get_output_factorizer

        if model is None:
            from causal_slr.model_training.train_model import load_weights, build_model

            conf.model_params.config_path = config_path
            model = build_model(
                conf, device=torch.device('cpu'), env=self.env)

            model = load_weights(model, conf, type='best')
            self.model_test = model
            print('Dynamics model loaded!\n')

        # with rendering function

        self.rollout_validation(model, demo_idxs=[70, 90], verbose=True)
        self.env.close()

    def rollout_world_model(self, state: torch.Tensor(), action: torch.Tensor()):
        input = rec_map_dict(lambda x: torch.Tensor(
            x).to(self.device).unsqueeze(0), state)
        action = torch.Tensor(action).to(self.device).unsqueeze(0)
        input['action'] = action
        outp = self.model_test(input)[0]
        outp = map_dict(lambda x: x.to(
            torch.device('cpu')).squeeze(0).numpy(), outp)

        return outp

    def rollout_validation(self, model, global_step=None, demo_idxs: list = [], verbose=False):
        losses_meter = RecursiveAverageMeter()

        self.model_test.load_state_dict(model.state_dict())
        self.device = model.device
        self.model_test.to(self.device)
        if self.model_test.training:
            self.model_test.eval()

        act, term, obs = self.data['actions'], self.data['terminals'], self.data['observations']
        # act, term, obs = self.data['actions'], self.data['terminals'], self.data['obs']
        indices = np.concatenate((np.array([-1]), np.where(term)[0]))
        demo_idxs = np.random.randint(
            len(indices)-1, size=10) if len(demo_idxs) == 0 else demo_idxs
        for demo_idx in demo_idxs:
            predictions = defaultdict(list)
            gt = defaultdict(list)
            start = indices[demo_idx]+1
            end = indices[demo_idx+1]

            num_rollout_steps = 6
            obs_fact = self.output_factorizer(obs[start])

            [predictions[k].append(v) for k, v in obs_fact.items()]
            [gt[k].append(v) for k, v in obs_fact.items()]

            with torch.no_grad():
                for action, observation, next_observation in zip(act[start:end-1], obs[start:end-1], obs[start+1:end]):
                    
                    if num_rollout_steps > 5: # Reset the state to groundtruth
                        obs_fact = self.input_factorizer(observation)
                        num_rollout_steps = 0
                    else:
                        obs_fact = pred_next_state


                    pred_next_state = self.rollout_world_model(
                        state=obs_fact, action=action)

                    [predictions[k].append(v)
                     for k, v in pred_next_state.items()]
                    [gt[k].append(v) for k, v in self.output_factorizer(
                        next_observation).items()]

                    losses = {}
                    for k in pred_next_state.keys():
                        error = (
                            (pred_next_state[k] - self.output_factorizer(next_observation)[k])**2).mean()
                        losses[f'val_rollout_{k}'] = AttrDict(
                            value=error, weight=1)
                        if verbose:
                            print(f'val_rollout_{k}: {error}')

                    losses_meter.update(losses)
                    del losses

                    num_rollout_steps += 1

                if verbose:
                    for k in predictions.keys():
                        self.plot_data(predictions[k], gt[k], k)
        # print('Rollout losses: ', losses_meter.avg,
        #       'global step: ', global_step)
        if not verbose:
            self.model_test.log_outputs(None, None, losses_meter.avg, global_step,
                                        log_images=False, phase='val_rollout')

    def plot_data(self, pred, gt, name,):
        plt.figure()
        plt.title(name)
        pred = np.asarray(pred)
        gt = np.asarray(gt)
        plt.plot(pred, label='pred', marker='.', color='red')
        plt.plot(gt, label='gt', marker='.', color='blue')

        plt.legend()
        plt.ylim(self.env.ELEMENT_LIMITS[name][1]+np.array([-0.5, 0.5]))
        plt.show()
