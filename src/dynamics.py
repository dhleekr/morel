import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from src.networks import DynamicsModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


class Dynamics:
    def __init__(self, env, dataset, args):
        self.args = args
        self.obs_dim = env.observation_space.shape[0]
        self.dataset = dataset
        self.threshold = self.args.threshold

        self.models = nn.ModuleList([DynamicsModel(env, args).to(device) for _ in range(self.args.num_models)])
        self.optimizers = [Adam(self.models[i].parameters(), lr=args.dynamics_lr) for i in range(args.num_models)]

        self.obs_mean = torch.Tensor(self.dataset.observation_mean).float().to(device)
        self.obs_std = torch.Tensor(self.dataset.observation_std).float().to(device)
        self.action_mean = torch.Tensor(self.dataset.action_mean).float().to(device)
        self.action_std = torch.Tensor(self.dataset.action_std).float().to(device)
        self.delta_mean = torch.Tensor(self.dataset.delta_mean).float().to(device)
        self.delta_std = torch.Tensor(self.dataset.delta_std).float().to(device)
        self.reward_mean = torch.Tensor([self.dataset.reward_mean]).float().to(device)
        self.reward_std = torch.Tensor([self.dataset.reward_std]).float().to(device)

        self.initial_obs_mean = torch.Tensor(self.dataset.initial_obs_mean).float().to(device)
        self.initial_obs_std = torch.Tensor(self.dataset.initial_obs_std).float().to(device)

        self.HALT_REWARD = -self.args.halt_penalty

        self.expert = Normal(self.action_mean, self.action_std)

        self.writer = SummaryWriter(f"./results/{self.args.env}")

    def forward(self, idx, obs):
        return self.models[idx](obs)

    def train(self, dataloader):
        print('\n')
        print('#'*50)
        print('Dynamics Training Starts...')
        print('#'*50)
        for epoch in range(self.args.dynamics_epochs):
            print(f"Epoch : {epoch + 1}")
            for idx, batch in enumerate(tqdm(dataloader)):
                feed, target = batch

                loss_vals = []

                for i in range(self.args.num_models):
                    predict = self.models[i](feed)
                    loss = F.mse_loss(predict, target)
                    loss_vals.append(loss)

                    self.optimizers[i].zero_grad()
                    loss.backward()
                    self.optimizers[i].step()

            print(f'Loss : {sum((l.item()) for l in loss_vals) / self.args.num_models}')
            print('\n')

            if self.args.save and epoch % self.args.save_freq == 0:
                self.save()
                
        print('#'*50)
        print('Dynamics Training Finished...')      
        print('#'*50)
        print('\n')

    def USAD(self, deltas):
        max_disc = torch.max(torch.cdist(deltas, deltas))
        return max_disc > self.threshold
        
    def predict(self, obs):
        with torch.no_grad():
            return torch.stack(list(map(lambda i: self.forward(i, obs), range(self.args.num_models))))
    
    def save(self):
        torch.save(self.models.state_dict(), f'./models/dynamics_{self.args.env}.pt')
        print('Model saved!!!!!!!!!!!!')

    def load(self):
        print('Model loading...')
        self.models.load_state_dict(torch.load(f'./models/dynamics_{self.args.env}.pt', map_location=device))
        self.threshold = torch.load(f'./models/threshold_{self.args.env}.pt', map_location=device)
        print(self.threshold)

    def get_max_disc(self, dataset):
        val = []
        observations = dataset.source_observation
        actions = dataset.source_action
        for i in tqdm(range(len(observations))):
            feed = torch.FloatTensor(np.concatenate([observations[i], actions[i]]))
            prediction = self.predict(feed).squeeze()
            delta = prediction[:, :self.obs_dim]
            val.append(torch.max(torch.cdist(delta, delta)))
        torch.save(max(val), f'./models/threshold_{self.args.env}.pt')
            


    """
    The functions below are for P-MDP
    """
    def step(self, state, action):
        action = (action - self.action_mean) / self.action_std
        state_normalized = (state - self.obs_mean) / self.obs_std
        predictions = self.predict(torch.cat([state_normalized, action], 0)).squeeze()
    
        deltas = predictions[:, :self.obs_dim]
        rewards = predictions[:, -1]

        unknown = self.USAD(deltas)

        deltas_unnormalized = self.delta_std * torch.mean(deltas, 0) + self.delta_mean
        next_obs =  state + deltas_unnormalized

        reward_out = self.reward_std * torch.mean(rewards) + self.reward_mean

        if(unknown):
            reward_out[0] = self.HALT_REWARD
        reward_out = torch.squeeze(reward_out)

        return next_obs, reward_out, unknown.item(), {"HALT" : unknown}

    def reset(self):
        idx = np.random.choice(self.dataset.initial_obs.shape[0])
        obs = torch.tensor(self.dataset.initial_obs[idx]).float().to(device)
        obs = obs * self.obs_std + self.obs_mean
        return obs