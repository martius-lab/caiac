from causal_slr.utils import conf_utils
from causal_slr.components.checkpointer import CheckpointHandler
from causal_slr.utils.general_utils import AttrDict, map_dict
import os
import numpy as np
import torch
from causal_slr.envs.kitchen_env import OBS_ELEMENT_INDICES


class FixedIntervalHierarchicalAgent():
    """Hierarchical agent that executes high-level actions in fixed temporal intervals."""

    def __init__(self, params, path):
        model = conf_utils.get_skill_model(
            params.skill_model_config.model_class)(params=params.skill_model_config)
        model.eval()
        self.load_weights(path, model)
        self.model = model
        self.hl_interval = params.skill_model_config.len_skill
        self._steps_since_hl = 0  # number of steps since last high-level step
        self.state_action_decode = params.skill_model_config.state_action_decode
        self.mpc_approach = params.skill_model_config.mpc_approach

    @torch.no_grad()
    def act(self, obs, task=None):
        """Output dict contains is_hl_step in case high-level action was performed during this action."""
        obs_input = obs[None] if len(
            obs.shape) == 1 else obs    # need batch input for agents

        output = AttrDict()
        if self._perform_hl_step_now:
            self.reset_hl_step_counter()
            obj_id = self.compute_obj_id(task)
            hl_input = self.get_hl_input(obs_input, obj_id)
            self._last_hl_output = self.model.compute_learned_prior(
                hl_input).sample()
            # if squash_output_dist:
            #     action, log_prob = self._tanh_squash_output(action, log_prob)
            output.is_hl_step = True
            if len(hl_input.shape) == 2 and len(self._last_hl_output.shape) == 1:
                # add batch dim if necessary
                self._last_hl_output = self._last_hl_output[None]
        else:
            output.is_hl_step = False
        output.last_hl_output = self._last_hl_output.numpy()
        # perform step with low-level policy
        assert self._last_hl_output is not None

        ll_agent_input = self.make_ll_obs(obs_input, self._last_hl_output)
        assert ll_agent_input.shape[1] == self.model.state_dim + \
            self.model.latent_dim, f'input to ll_agent {ll_agent_input.shape} is different than state_dim + latnent_dim {self.model.state_dim} + {self.model.latent_dim}'
        ll_agent_input = torch.Tensor(ll_agent_input, device=self.model.device)
        output.ll_action = self.model.decoder(ll_agent_input).numpy()
        self._steps_since_hl += 1
        return self._remove_batch(output) if len(obs.shape) == 1 else output

    def get_hl_input(self, obs_input, obj_id):
        hl_input = np.concatenate((obs_input, obj_id), axis=-1)
        assert hl_input.shape[1] == self.model.prior_input_size
        # perform step with high-level policy
        hl_input = torch.Tensor(hl_input, device=self.model.device)
        assert hl_input.shape[1] == self.model.prior_input_size
        return hl_input

    def make_ll_obs(self, obs, hl_action):
        """Creates low-level agent's observation from env observation and HL action.
        returns: np.array([state, z])"""
        if not self.model.goal_conditioned:
            return np.concatenate((obs, hl_action), axis=-1)
        else:
            # need batch input for agents
            obs = obs[None] if len(obs.shape) == 1 else obs
            hl_action = hl_action[None] if len(
                hl_action.shape) == 1 else hl_action
            return np.concatenate((obs[:, :-self.model.goal_dim], hl_action), axis=-1)

    def compute_obj_id(self, task):
        id = OBS_ELEMENT_INDICES[task][0]
        robot_id = OBS_ELEMENT_INDICES['robot'][0]
        id_task_robot = [id, robot_id]
        return self.multilabel_encoder.transform([id_task_robot])

    def reset_hl_step_counter(self):
        self._steps_since_hl = 0     # start new episode with high-level step

    @property
    def _perform_hl_step_now(self):
        if self.mpc_approach:
            return True
        else:
            return self._steps_since_hl % self.hl_interval == 0

    def load_weights(self, weights_path, model):
        checkpoint_dir = weights_path if os.path.basename(weights_path) == 'weights' \
            else os.path.join(weights_path, 'weights')
        checkpoint_path = CheckpointHandler.get_resume_ckpt_file(
            'latest', checkpoint_dir)
        CheckpointHandler.load_weights(checkpoint_path, model)

    @staticmethod
    def _remove_batch(d):
        """Remove batch dimension to all tensors in d."""
        return map_dict(lambda x: x[0] if (isinstance(x, torch.Tensor) or
                                           isinstance(x, np.ndarray)) else x, d)


class FixedIntervalHierarchicalFullDecodeAgent(FixedIntervalHierarchicalAgent):
    """Hierarchical agent that executes high-level actions in fixed temporal intervals."""

    def __init__(self, params, path):
        super().__init__(params, path)

    @torch.no_grad()
    def act(self, obs, task=None):
        """Output dict contains is_hl_step in case high-level action was performed during this action."""
        obs_input = obs[None] if len(
            obs.shape) == 1 else obs    # need batch input for agents

        output = AttrDict()
        if self._perform_hl_step_now:
            self.reset_hl_step_counter()
            # Modify obs_input with heuristic obj_id for now
            obj_id = self.compute_obj_id(task)
            hl_input = self.get_hl_input(obs_input, obj_id)
            self._last_hl_output = self.model.compute_learned_prior(
                hl_input).sample()
            output.is_hl_step = True
            if len(hl_input.shape) == 2 and len(self._last_hl_output.shape) == 1:
                # add batch dim if necessary
                self._last_hl_output = self._last_hl_output[None]
        else:
            output.is_hl_step = False
        output.last_hl_output = self._last_hl_output.numpy()
        # perform step with low-level policy
        assert self._last_hl_output is not None

        # we only need to decode full trajectory if z changes (since it doesnt depend on s)
        if self._perform_hl_step_now:
            self._last_ll_output = self.model.decoder(
                self._last_hl_output).numpy()
        # Otherwise we can just the last decoded trajectory
        output.full_decode = self._last_ll_output

        output.ll_action = self.get_action(output.full_decode)
        self._steps_since_hl += 1
        return output

    def get_action(self, full_decode):
        if self.state_action_decode:
            actions = full_decode[:, self.model.state_dim *
                                  self.hl_interval::].reshape(self.hl_interval, -1)
        else:
            actions = full_decode.reshape(self.hl_interval, -1)

        action = actions[self._steps_since_hl, :]
        return action


class FixedIntervalHierarchicalAgentBaseline(FixedIntervalHierarchicalAgent):
    """Hierarchical agent that executes high-level actions in fixed temporal intervals."""

    def __init__(self, params, path):
        super().__init__(params, path)

    def get_hl_input(self, hl_input, task=None):
        hl_input = torch.Tensor(hl_input, device=self.model.device)
        return hl_input

    def compute_obj_id(self, task):
        return None


class FixedIntervalHierarchicalFullDecodeAgentBaseline(FixedIntervalHierarchicalFullDecodeAgent):
    """Hierarchical agent that executes high-level actions in fixed temporal intervals."""

    def __init__(self, params, path):
        super().__init__(params, path)

    def get_hl_input(self, hl_input, task=None):
        hl_input = torch.Tensor(hl_input, device=self.model.device)
        return hl_input

    def compute_obj_id(self, task):
        return None
