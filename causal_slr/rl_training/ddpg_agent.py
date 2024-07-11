import torch
import numpy as np
from .base_agent import BaseAgent
from .models import Actor, QValue


class DDPG(BaseAgent):

    def __init__(self, args, env):
        super().__init__(args, env) 
        self.actor_network = Actor(self.args).to(self.device)
        self.critic_network = QValue(self.args).to(self.device)
        self.actor_target_network = Actor(self.args).to(self.device)
        self.critic_target_network = QValue(self.args).to(self.device)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

    def _deterministic_action(self, input_tensor):
        action = self.actor_network(input_tensor)
        return action
    
    def _stochastic_action(self, input_tensor):
        action = self._deterministic_action(input_tensor)
        epsilon = torch.randn_like(action) * self.args.act_noise
        action = torch.clamp(action + epsilon, -self.args.env_action_max, self.args.env_action_max)
        return action

    def _optimize(self, loss, optim):
        optim.zero_grad()
        loss.backward()
        optim.step()

    def _get_training_batch(self, future_p=None):
        sample_batch = self.sample_batch(future_p=future_p)
        transitions = sample_batch['transitions']
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        to_tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=self.device)
        return (to_tensor(x) for x in [inputs_norm, inputs_next_norm, transitions['actions'], transitions['r']])

    def _update_network(self, future_p=None):

        state, next_state, action, reward = self._get_training_batch(future_p=future_p)

        with torch.no_grad():
            next_action = self.actor_target_network(next_state)
            q_next_state = self.critic_target_network(next_state, next_action)
            target_q = reward + self.args.gamma * q_next_state
            clip_return = 1 / (1 - self.args.gamma)
            target_q = torch.clamp(target_q, -clip_return, 0).detach()

        q_hat = self.critic_network(state, action)
        critic_loss = (target_q - q_hat).pow(2).mean()
        pi = self.actor_network(state)
        actor_loss = -self.critic_network(state, pi).mean()
        actor_loss += self.args.action_l2 * (pi / self.args.env_action_max).pow(2).mean()

        self._optimize(actor_loss, self.actor_optim)
        self._optimize(critic_loss, self.critic_optim)
        return {'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item(),
                'q_hat': q_hat.mean().item()}
