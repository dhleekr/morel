import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


def weights_init(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)
        module.bias.data.fill_(0.01)


class DynamicsModel(nn.Module):
    def __init__(self, env, args):
        super(DynamicsModel, self).__init__()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        # Output : state + reward
        self.model = MLP(input_dim=obs_dim + act_dim, output_dim=obs_dim + 1, hidden_dim=args.dynamics_hidden_dim)
    
    def forward(self, obs):
        return self.model(obs)


class Actor(nn.Module):
    def __init__(self, env, args):
        super(Actor, self).__init__()
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.mlp = MLP(input_dim=self.obs_dim, output_dim=args.policy_hidden_dim[-1], hidden_dim=args.policy_hidden_dim)
        self.mean_layer = nn.Linear(args.policy_hidden_dim[-1], self.act_dim)
        self.log_std_layer = nn.Linear(args.policy_hidden_dim[-1], self.act_dim)

        self.apply(weights_init)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        out = self.mlp(obs)
        mean = self.mean_layer(out)
        log_std = self.log_std_layer(out)
        log_std = torch.clamp(log_std, min=-2, max=5)
        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, torch.tensor(self.env.action_space.low), torch.tensor(self.env.action_space.high))
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob
    
    def evaluate_action(self, obs, action):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        print(obs, action)
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