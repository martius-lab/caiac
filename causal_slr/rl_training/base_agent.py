from causal_slr.utils.general_utils import save_config
import hashlib
import yaml
import glob
from smart_settings.param_classes import recursive_objectify
from causal_slr.utils.general_utils import get_data_dir
from causal_slr.utils.rl_utils import discounted_return, update_avg_dict, Logger
from causal_slr.utils.rl_utils import Normalizer
from causal_slr.her_modules.her import HERSampler
from .replay_buffer import ReplayBuffer
import torch
import numpy as np
import imageio
from tqdm import tqdm
import os


PREFIX = os.getcwd()
DATA_PATH = PREFIX + '/data'
MAX_RUNTIME = 60 * 60 * 5


class BaseAgent:

    def __init__(self, args, env):
        os.environ['DATA_DIR'] = DATA_PATH
        self.args = args
        self.env = env
        self.device = 'cuda' if args.cuda else 'cpu'

        self.model_path = os.path.join(self.args.working_dir, 'checkpoints')
        self.gif_path = os.path.join(self.args.working_dir, 'gifs')
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.gif_path, exist_ok=True)
        save_config(args, os.path.join(self.args.working_dir, 'config.yaml'))
        self.logger = Logger(self.args)

        self.model_cai = self.load_model_cai()
        self.her_sampler = HERSampler(self.args.replay_strategy, self.args.relabel_percent,
                                      self.env.compute_reward)
        if self.args.expert_percent == self.args.random_percent == 0:  # TODO make this nicer
            # Here we are using the augmented caiac datsett which is already mixed
            self.buffer = CAIACDYNA_ReplayBuffer(self.args, self.args.buffer_size, self.her_sampler.sample_her_transitions, model_cai=self.model_cai,
                                                 reward_func=self.env.compute_reward, env_obj_2_idx=self.env.OBJ_2_IDX, env_obs_2_ag=self.env.obs_2_ag,
                                                 env_input_factorizer=self.env.get_input_factorizer)
        else:
            self.buffer = ReplayBuffer(self.args, self.args.buffer_size, self.her_sampler.sample_her_transitions, model_cai=self.model_cai,
                                       reward_func=self.env.compute_reward, env_obj_2_idx=self.env.OBJ_2_IDX, env_obs_2_ag=self.env.obs_2_ag,
                                       env_input_factorizer=self.env.get_input_factorizer)

        self.o_norm = Normalizer(
            size=self.args.env_obs_size, default_clip_range=self.args.clip_range)
        self.g_norm = Normalizer(
            size=self.args.env_goal_size, default_clip_range=self.args.clip_range)

    def _preprocess_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        inputs = np.concatenate([obs_norm, g_norm]) if len(
            obs_norm.shape) == 1 else np.concatenate([obs_norm, g_norm], axis=1)
        inputs = torch.tensor(inputs, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        return inputs

    def _soft_update(self):
        self._soft_update_target_network(
            self.actor_target_network, self.actor_network)
        self._soft_update_target_network(
            self.critic_target_network, self.critic_network)

    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def _update_network(self):
        pass

    def _update_discriminator(self):
        pass

    def sample_batch(self, future_p=None):
        sample_batch = self.buffer.sample(
            self.args.batch_size, future_p=future_p, agent=self)
        transitions = sample_batch['transitions']
        def clip_to_range(x): return np.clip(
            x, -self.args.clip_obs, self.args.clip_obs)
        for k in ['obs', 'obs_next', 'g']:
            transitions[k] = clip_to_range(transitions[k])
        transitions['g_next'] = transitions['g']
        sample_batch['transitions'] = transitions
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()
        return sample_batch

    @torch.no_grad()
    def _eval_and_save_agent(self, make_gif=False, epoch=0, test_rollouts=None, save=False):

        record_keys = ['ag', 'g', 'rew', 'sr']
        all = {k: [] for k in record_keys}
        num_test_rollouts = test_rollouts if test_rollouts is not None else self.args.n_test_rollouts
        for n_roll in range(num_test_rollouts):

            single = {k: [] for k in record_keys}
            imgs = []
            observation = self.env.reset()
            obs, ag, g = (observation[k] for k in [
                          'observation', 'achieved_goal', 'desired_goal'])
            for _ in range(self.args.env_max_timesteps):

                input_tensor = self._preprocess_inputs(obs, g)
                actions = self._deterministic_action(input_tensor)
                actions = actions.detach().cpu().numpy().squeeze()

                new_observation, reward, _, info = self.env.step(actions)

                success = float(
                    info['score/success' if 'score/success' in info else 'is_success'])
                imgs.append(self.env.render(mode='rgb_array')
                            if make_gif else None)
                [single[k].append(v) for k, v in zip(
                    record_keys, [ag, g, reward, success])]
                obs, ag, g = (new_observation[k] for k in [
                              'observation', 'achieved_goal', 'desired_goal'])

            [all[k].append(single[k]) for k in record_keys]
            if make_gif:
                imageio.mimsave(
                    self.gif_path + f'/e{epoch}_t{n_roll}.gif', np.array(imgs))

        all = {k: np.array(v) for k, v in all.items()}
        dis_return, undis_return = discounted_return(
            all['rew'], self.args.gamma)
        if save:
            self.save(self.model_path + f'/e{epoch}.pt')
        results = {
            'epoch': epoch,
            'final_distance': float(np.mean(np.linalg.norm(all['ag'][:, -1] - all['g'][:, -1], axis=1))),
            'final_success_rate': float(np.mean(all['sr'][:, -1])),
            'success_rate': float(np.mean(all['sr'].any(1))),
            'discounted_return': float(np.mean(dis_return)),
            'undiscounted_return': float(np.mean(undis_return)),
        }
        return results

    def save(self, path):
        torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean,
                   self.g_norm.std, self.actor_network, self.critic_network1], path)

    def learn(self):

        load_path_expert = get_data_dir(
        ) + f'/expert/{self.args.env[:-3]}/buffer.pkl'
        load_path_random = get_data_dir(
        ) + f'/random/{self.args.env[:-3]}/buffer.pkl'
        self.buffer.load_mixture(load_path_expert, load_path_random, self.args.expert_percent,
                                 self.args.random_percent, self.args.shrink_dataset, env_name=self.args.env)
        future_p = self.her_sampler.future_p
        if self.args.ratio_cf > 0.:
            self.buffer.compute_cais(future_p)
        for epoch in range(self.args.n_epochs):
            save = True if not epoch % 200 else False
            env_metrics = self._eval_and_save_agent(epoch=epoch, save=save)
            avg_metrics = {}
            future_p = self.her_sampler.future_p
            for _ in tqdm(range(self.args.n_cycles)):
                for _ in range(self.args.n_batches):
                    train_metrics = self._update_network(future_p=future_p)
                    update_avg_dict(avg_metrics, train_metrics)
                self._soft_update()

            avg_metrics = {k: np.mean(v) for k, v in avg_metrics.items()}
            self.logger.log({**env_metrics, **avg_metrics})
            print(
                '\t'.join([k + f': {v:.3f}' for k, v in env_metrics.items()]))

        env_metrics = self._eval_and_save_agent(
            make_gif=False, epoch=epoch, test_rollouts=50, save=True)
        self.logger.log_hparams({**env_metrics, **avg_metrics})
        self.logger.close()
        return env_metrics

    def load_model_cai(self):
        from causal_slr.model_training.train_model import load_weights, build_model
        filelist = glob.glob(f'{self.args.mdp_config_path}/*.yaml')
        if len(filelist) == 1:
            config_path = filelist[0]
        else:
            raise FileNotFoundError(
                f'Couldnt find world model file {self.args.mdp_config_path}')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        config = recursive_objectify(config, make_immutable=False)
        config.model_params.config_path = config_path
        print('\nBuilding dynamics model cai ...')
        model = build_model(config, env=self.env)

        # this does the model.to(self.device) already
        model.device = self.device
        model = load_weights(model, config, type='best')
        print('Dynamics model loaded!\n')
        # Saving hash of model file to working_dir
        with open(model.weights_file, 'rb') as inputfile:
            hash = hashlib.md5(inputfile.read()).hexdigest()
        with open(os.path.join(self.args.working_dir, 'model_hash.txt'), 'w') as outputfile:
            print(f'Saving hash for model in ', self.args.working_dir)
            outputfile.write(hash)

        return model

    def get_augmented_data(self):

        load_path_expert = get_data_dir(
        ) + f'/expert/{self.args.env[:-3]}/buffer.pkl'
        load_path_random = get_data_dir(
        ) + f'/random/{self.args.env[:-3]}/buffer.pkl'
        self.buffer.load_mixture(load_path_expert, load_path_random, self.args.expert_percent,
                                 self.args.random_percent, self.args.shrink_dataset, env_name=self.args.env)
        future_p = self.her_sampler.future_p
        self.buffer.compute_cais(future_p)

        return self.buffer.cf_buffer, self.buffer.cf_buffer_size
