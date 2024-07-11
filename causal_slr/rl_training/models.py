import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.max_action = args.env_action_max
        self.fc1 = nn.Linear(args.env_obs_size + args.env_goal_size, args.units)
        self.fc2 = nn.Linear(args.units, args.units)
        self.fc3 = nn.Linear(args.units, args.units)
        self.action_out = nn.Linear(args.units, args.env_action_size)
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions


class QValue(nn.Module):
    def __init__(self, args, activation=None):
        super(QValue, self).__init__()
        self.activation = activation
        self.max_action = args.env_action_max
        self.fc1 = nn.Linear(args.env_obs_size + args.env_goal_size + args.env_action_size, args.units)
        self.fc2 = nn.Linear(args.units, args.units)
        self.fc3 = nn.Linear(args.units, args.units)
        self.q_out = nn.Linear(args.units, 1)

        self.apply(weights_init_)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        if self.activation == 'sigmoid':
            q_value = torch.sigmoid(q_value)
        return q_value

class DQValue(nn.Module):
    def __init__(self, args, activation=None):
        super(DQValue, self).__init__()
        self.activation = activation
        self.max_action = args.env_action_max
        self.fc1 = nn.Linear(args.env_obs_size + args.env_goal_size + args.env_action_size, args.units)
        self.fc2 = nn.Linear(args.units, args.units)
        self.fc3 = nn.Linear(args.units, args.units)
        self.parameterization = args.parameterization
        if args.spectral:
            self.fc1 = torch.nn.utils.spectral_norm(self.fc1)
            self.fc2 = torch.nn.utils.spectral_norm(self.fc2)
            self.fc3 = torch.nn.utils.spectral_norm(self.fc3)
        if self.parameterization == 'tree':
            self.q_out = nn.Linear(args.units, 63)
            self._compute_w_matrices()
        else:
            self.q_out = nn.Linear(args.units, 50)
        self.apply(weights_init_)

    def _compute_w_matrices(self):
        depth = 5
        n_leafs = 2**(depth+1) - 2
        pos_w = torch.zeros((n_leafs+1, n_leafs+2))
        neg_w = torch.zeros((n_leafs+1, n_leafs+2))
        n_leafs = 2*n_leafs
        tree = {0: {'children': [], 'left_par': [], 'right_par': []}}
        free_id = 0
        for i in range(n_leafs+2):
            if len(tree[free_id]['children']) >= 2:
                free_id += 1
            tree[free_id]['children'].append(i+1)
            tree[i+1] = {'children': [],
                        'left_par': tree[free_id]['left_par'].copy(),
                        'right_par': tree[free_id]['right_par'].copy()}
            if len(tree[free_id]['children']) == 1:
                tree[i+1]['left_par'] += [free_id]
            else:
                tree[i+1]['right_par'] += [free_id]

            if i+2 >= (2**(depth+1)):
                for p in tree[i+1]['left_par']:
                    pos_w[p][i+2-(2**(depth+1))] = 1.
                for p in tree[i+1]['right_par']:
                    neg_w[p][i+2-(2**(depth+1))] = 1.
            self.w_p = pos_w[None, ...]
            self.w_q = neg_w[None, ...]

    def forward(self, x, actions, return_logits=False):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.q_out(x)
        if self.parameterization == 'softmax':
            q_value = torch.softmax(logits, dim=-1)
        elif self.parameterization == 'mixture':
            weight = torch.nn.functional.sigmoid(logits[:, [0]])
            q_value = torch.nn.functional.softmax(logits[:, 1:], -1)
            q_value = torch.cat([weight, (1-weight)*q_value], -1)
        elif self.parameterization == 'tree':
            q_value = torch.nn.functional.sigmoid(logits)[..., None]
            q_value = q_value * self.w_p + (1-q_value) * self.w_q
            q_value = torch.where(self.w_p.abs() + self.w_q.abs() == 0, 1., q_value)
            q_value = q_value.prod(-2)
        else:
            raise NotImplementedError(f'Unknown parameterization: {self.parameterization}.')
        return (q_value, logits) if return_logits else q_value

class Value(nn.Module):
    def __init__(self, args):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(args.env_obs_size + args.env_goal_size, args.units)
        self.fc2 = nn.Linear(args.units, args.units)
        self.fc3 = nn.Linear(args.units, args.units)
        self.q_out = nn.Linear(args.units, 1)
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value

class DValue(nn.Module):
    def __init__(self, args, activation=None):
        super(DValue, self).__init__()
        self.activation = activation
        self.max_action = args.env_action_max
        self.fc1 = nn.Linear(args.env_obs_size + args.env_goal_size, args.units)
        self.fc2 = nn.Linear(args.units, args.units)
        self.fc3 = nn.Linear(args.units, args.units)
        if args.spectral:
            self.fc1 = torch.nn.utils.spectral_norm(self.fc1)
            self.fc2 = torch.nn.utils.spectral_norm(self.fc2)
            self.fc3 = torch.nn.utils.spectral_norm(self.fc3)
        self.q_out = nn.Linear(args.units, 50)
        self.apply(weights_init_)


    def forward(self, x, return_logits=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.q_out(x)
        q_value = torch.softmax(logits, dim=-1)
        return (q_value, logits) if return_logits else q_value


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/gail.py
class Discriminator(nn.Module):
    def __init__(self, input_dim, units, lr=5e-4):
        super(Discriminator, self).__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, units), nn.Tanh(),
            nn.Linear(units, units), nn.Tanh(),
            nn.Linear(units, units), nn.Tanh(),
            nn.Linear(units, 1))
        self.trunk.train()
        self.optimizer = torch.optim.Adam(self.trunk.parameters(), lr=5e-4)

    def compute_grad_pen(self, expert_data, offline_data, lambda_=20.):
        alpha = torch.rand(expert_data.size(0), 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)
        mixup_data = alpha * expert_data + (1 - alpha) * offline_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    @torch.no_grad()
    def predict_reward(self, state):
        d = self.trunk(state)
        s = torch.sigmoid(d)
        reward = s.log() - (1 - s).log()
        return reward 
