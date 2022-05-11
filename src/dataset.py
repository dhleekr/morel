
import numpy as np
import torch
from torch.utils.data import Dataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

class OfflineDataset(Dataset):
    def __init__(self, env):
        self.env = env
        dataset = self.env.get_dataset()

        # Input data
        self.source_observation = dataset["observations"][:-1]
        self.source_action = dataset["actions"][:-1]

        # Output data
        self.target_delta = dataset["observations"][1:] - self.source_observation
        self.target_reward = dataset["rewards"][:-1]

        # Normalize data
        self.delta_mean = self.target_delta.mean(axis=0)
        self.delta_std = self.target_delta.std(axis=0)

        self.reward_mean = self.target_reward.mean(axis=0)
        self.reward_std = self.target_reward.std(axis=0)

        self.observation_mean = self.source_observation.mean(axis=0)
        self.observation_std = self.source_observation.std(axis=0)

        self.action_mean = self.source_action.mean(axis=0)
        self.action_std = self.source_action.std(axis=0)

        self.source_action = (self.source_action - self.action_mean) / self.action_std
        self.source_observation = (self.source_observation - self.observation_mean) / self.observation_std
        self.target_delta = (self.target_delta - self.delta_mean) / self.delta_std
        self.target_reward = (self.target_reward - self.reward_mean) / self.reward_std

        # Get indices of initial states
        self.done_indices = dataset["timeouts"][:-1]
        self.initial_indices = np.roll(self.done_indices, 1)
        self.initial_indices[0] = True

        # Calculate distribution parameters for initial states
        self.initial_obs = self.source_observation[self.initial_indices]
        self.initial_obs_mean = self.initial_obs.mean(axis = 0)
        self.initial_obs_std = self.initial_obs.std(axis = 0)

        # Remove transitions from terminal to initial states
        self.source_action = np.delete(self.source_action, self.done_indices, axis = 0)
        self.source_observation = np.delete(self.source_observation, self.done_indices, axis = 0)
        self.target_delta = np.delete(self.target_delta, self.done_indices, axis = 0)
        self.target_reward = np.delete(self.target_reward, self.done_indices, axis = 0)

    def __getitem__(self, idx):
        feed = torch.FloatTensor(np.concatenate([self.source_observation[idx], self.source_action[idx]])).to(device)
        target = torch.FloatTensor(np.concatenate([self.target_delta[idx], self.target_reward[idx:idx+1]])).to(device)
        return feed, target

    def __len__(self):
        return len(self.source_observation)