import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def weights_init(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)
        module.bias.data.fill_(0.01)

class DynamicsModel(nn.Module):
    def __init__(self, env, mu_s, mu_a, std_s, std_a, std_delta, args):
        super(DynamicsModel, self).__init__()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.model = MLP(input_dim=obs_dim + act_dim, output_dim=args.dynamics_hidden_dim[-1], hidden_dim=args.dynamics_hidden_dim)
        self.mean_layer = nn.Linear(args.dynamics_hidden_dim[-1], obs_dim)
        self.log_std_layer = nn.Linear(args.dynamics_hidden_dim[-1], obs_dim)
        
        self.mu_s = mu_s
        self.mu_a = mu_a
        self.std_s = std_s
        self.std_a = std_a
        self.std_delta = std_delta

        self.apply(weights_init)

    def forward(self, state, action):
        s_mlp = (state - self.mu_s) / (self.std_s + 1e-7)
        a_mlp = (action - self.mu_a) / (self.std_a + 1e-7)
        out = self.model(s_mlp, a_mlp)
        mean = state + self.std_delta * self.mean_layer(out)
        log_std = self.log_std_layer(out)
        log_std = torch.clamp(log_std, -20, 5)
        return mean, log_std

    def get_next_state(self, state, action):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        mean, log_std = self.forward(state, action)
        std = log_std.exp()
        dist = Normal(mean, std)
        return dist.rsample()


class Actor(nn.Module):
    def __init__(self, env, args):
        super(Actor, self).__init__()
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.mlp = MLP(input_dim=self.obs_dim, output_dim=args.policy_hidden_dim[-1], hidden_dim=args.policy_hidden_dim, activation_fn=nn.Tanh())
        self.mean_layer = nn.Linear(args.policy_hidden_dim[-1], self.act_dim)
        self.log_std_layer = nn.Linear(args.policy_hidden_dim[-1], self.act_dim)

        self.apply(weights_init)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        out = self.mlp(obs)
        mean = self.mean_layer(out)
        log_std = self.log_std_layer(out)
        log_std = torch.clamp(log_std, -20, 5)
        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob
    
    def evaluate_action(self, obs, action):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = action.view(-1, self.act_dim)
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob, dist.entropy()


class Critic(nn.Module):
    def __init__(self, env, args):
        super(Critic, self).__init__()
        obs_dim = env.observation_space.shape[0]
        self.mlp = MLP(input_dim=obs_dim, output_dim=1, hidden_dim=args.value_hidden_dim)

        self.apply(weights_init)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        return self.mlp(obs)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[], activation_fn=nn.ReLU()):
        super(MLP, self).__init__()
        self.output_dim = output_dim

        modules = []
        prev_dim = input_dim
        for dim in hidden_dim:
            modules.append(nn.Linear(prev_dim, dim))
            modules.append(activation_fn)
            prev_dim = dim
        modules.append(nn.Linear(prev_dim, output_dim))
        self.fc_layers = nn.Sequential(*modules)

    def forward(self, obs, action=None):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float)
        if action != None:
            if len(action.shape) == 1:
                action = action.unsqueeze(0)
            out = torch.cat([obs, action], dim=1)
        else:
            out = obs
        return self.fc_layers(out)