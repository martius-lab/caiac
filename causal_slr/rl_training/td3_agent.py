import itertools
import torch
from .base_agent import BaseAgent
from .ddpg_agent import DDPG
from .models import Actor, QValue
import torch.nn.functional as F


class TD3(DDPG):

    def __init__(self, args, env):
        BaseAgent.__init__(self, args, env) 
        self.actor_network = Actor(self.args).to(self.device)
        self.critic_network1 = QValue(self.args).to(self.device)
        self.critic_network2 = QValue(self.args).to(self.device)
        self.actor_target_network = Actor(self.args).to(self.device)
        self.critic_target_network1 = QValue(self.args).to(self.device)
        self.critic_target_network2 = QValue(self.args).to(self.device)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network1.load_state_dict(self.critic_network1.state_dict())
        self.critic_target_network2.load_state_dict(self.critic_network2.state_dict())
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        critic_params = itertools.chain(self.critic_network1.parameters(), self.critic_network2.parameters())
        self.critic_optim = torch.optim.Adam(critic_params, lr=self.args.lr_critic)
        self.timer = 0

    def _soft_update(self):
        self._soft_update_target_network(self.actor_target_network, self.actor_network)
        self._soft_update_target_network(self.critic_target_network1, self.critic_network1)
        self._soft_update_target_network(self.critic_target_network2, self.critic_network2)

    def _update_network(self, future_p=None):

        state, next_state, action, reward = self._get_training_batch(future_p=future_p)

        with torch.no_grad():
            next_action = self.actor_target_network(next_state)
            epsilon = torch.randn_like(next_action) * self.args.target_noise
            epsilon = torch.clamp(epsilon, -self.args.noise_clip, self.args.noise_clip)
            next_action = torch.clamp(next_action + epsilon, -self.args.env_action_max, self.args.env_action_max)

            q_next_state1 = self.critic_target_network1(next_state, next_action)
            q_next_state2 = self.critic_target_network2(next_state, next_action)
            q_next_state = torch.min(q_next_state1, q_next_state2)
            target_q = reward + self.args.gamma * q_next_state
            clip_return = 1 / (1 - self.args.gamma)
            target_q = torch.clamp(target_q, -clip_return, 0).detach()

        q_hat1 = self.critic_network1(state, action)
        q_hat2 = self.critic_network2(state, action)
        critic_loss1 = (target_q - q_hat1).pow(2).mean()
        critic_loss2 = (target_q - q_hat2).pow(2).mean()
        critic_loss = critic_loss1 + critic_loss2
        train_metrics = {'critic_loss': critic_loss.item(), 'q_hat': (q_hat1+q_hat2).mean().item()/2}

        if self.timer % self.args.policy_delay == 0:
            actor_loss = self.compute_actor_loss(state, action=action)
            train_metrics.update({'actor_loss': actor_loss.item()})
            self._optimize(actor_loss, self.actor_optim)
        self.timer += 1
        self._optimize(critic_loss, self.critic_optim)

        return train_metrics
    
    def compute_actor_loss(self, state, **kwargs):
        pi = self.actor_network(state)
        actor_loss = -self.critic_network1(state, pi).mean()
        actor_loss += self.args.action_l2 * (pi / self.args.env_action_max).pow(2).mean()
        return actor_loss
        


class TD3_BC(TD3):
    def __init__(self, args, env):
        super().__init__(args, env)
        self.alpha_bc = args.alpha_bc # alpha --> 1 resembles BC, alpha --> 4 resembles RL

    def compute_actor_loss(self, state,action):
        pi = self.actor_network(state)
        Q = self.critic_network1(state, pi)
        lmbda = self.alpha_bc / Q.abs().mean().detach()
        
        actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)
        actor_loss += self.args.action_l2 * (pi / self.args.env_action_max).pow(2).mean()
        return actor_loss

        