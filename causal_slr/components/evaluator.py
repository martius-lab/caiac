import glob
import numpy as np

import yaml
from causal_slr.utils.general_utils import AttrDict, timing, recursive_objectify
from causal_slr.components.agent import FixedIntervalHierarchicalAgentBaseline, FixedIntervalHierarchicalFullDecodeAgentBaseline
from causal_slr.utils.general_utils import set_seeds

from causal_slr.utils.general_utils import RolloutStorage
from causal_slr.utils import conf_utils
from causal_slr.components.rollout import RolloutGenerator
from causal_slr.utils.vis_utils import add_captions_to_seq


class InferenceEvaluator:

    def __init__(self, args, logger):
        self.args = args
        self.conf = self.get_config()
        set_seeds(self.conf.seed)
        # build env
        self.conf.env_config.seed = self.conf.seed
        self.object_centric = self.conf.skill_general.with_cai
        self.env = conf_utils.get_env(
            self.conf.env_config.name, self.conf.env_config)
        print('Setting seed env', self.args.seed)
        self.env.seed(self.args.seed)
        self.logger = logger  # pass logger from skill learning

    def get_agent(self):
        if not self.conf.skill_model_config.full_decode:
            agent = FixedIntervalHierarchicalAgentBaseline(
                self.conf, path=self.args.skill_config_path)

        else:
            agent = FixedIntervalHierarchicalFullDecodeAgentBaseline(
                self.conf, path=self.args.skill_config_path)

        return agent

    def task_evaluation(self, step, render=False, render_mode='rgb_array', num_vals=None):
        agent = self.get_agent()

        roll_generator = RolloutGenerator(
            agent, self.env, self.conf.env_config.max_episode_len)
        val_rollout_storage = RolloutStorage()
        if num_vals is None:
            num_vals = self.args.skill_general.n_episodes_inference_val

        print('Evaluating {} episodes' .format(num_vals))
        with timing("Eval rollout time: "):
            for i in range(num_vals):
                episode = roll_generator.sample_episode(
                    render=render, render_mode=render_mode)
                val_rollout_storage.append(episode)
                # print(episode['task'], episode['ep_success'])

        rollout_stats = val_rollout_storage.rollout_stats()
        print(rollout_stats)
        if not self.args.dont_save and self.logger:
            self.log_outputs(rollout_stats, val_rollout_storage,
                             log_videos=self.args.log_videos, step=step, prefix='val_task')

        print("Evaluation Success_Rate: {}".format(
            rollout_stats.success_rate))
        del val_rollout_storage
        self.env.close()
        return rollout_stats

    def get_config(self):
        filelist = glob.glob(f'{self.args.skill_config_path}/*.yaml')
        if len(filelist) == 1:
            skill_config_path = filelist[0]
        else:
            raise FileNotFoundError(
                f'Couldnt find skill agent path {self.args.skill_config_path}')
        with open(skill_config_path, 'r') as file:
            skill_config = yaml.safe_load(file)
        skill_config = recursive_objectify(AttrDict(skill_config))
        return skill_config

    def log_outputs(self, logging_stats, rollout_storage, log_videos, step, prefix=None):
        """Logs all training outputs."""

        self.logger.log_scalar_dict(logging_stats, prefix=prefix, step=step)

        if log_videos:
            assert rollout_storage is not None, 'No rollout data available for image logging'
            # log rollout videos with info captions
            if 'image' in rollout_storage and rollout_storage.get()[0].image[0] is not None:
                if self.args.log_video_caption:
                    # rollout.info is a list of dicts, each dict has 'tasks_to_complete' and 'success' keys, which will be added as caption
                    vids = [np.stack(add_captions_to_seq(rollout.image, rollout.add_info)).transpose(0, 3, 1, 2)
                            for rollout in rollout_storage.get()[-self.logger.n_logged_samples:]]
                else:
                    vids = [np.stack(rollout.image).transpose(0, 3, 1, 2)
                            for rollout in rollout_storage.get()[-self.logger.n_logged_samples:]]  # Only record videos for the last n_logged_samples episodes
                self.logger.log_videos(vids, name="rollouts", step=step)
            else:
                print("No image key in rollout_storage for image logging!")
                pass

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
        config = recursive_objectify(AttrDict(config))
        config.model_params.config_path = config_path
        print('\nBuilding dynamics model cai ...')
        model = build_model(config)
        model = load_weights(model, config, type='best')
        print('Dynamics model loaded!\n')
        return model
