from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from .networks import DynamicsModel
from .replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


class Dynamics:
    def __init__(self, env, dataset, args):
        self.env = env
        self.dataset = dataset
        self.args = args
        self.obs_dim = env.observation_space.shape[0]

        self.mu_s = dataset['observations'].mean()
        self.mu_a = dataset['actions'].mean()
        self.std_s = dataset['observations'].std()
        self.std_a = dataset['actions'].std()
        self.std_delta = (dataset['next_observations'] - dataset['observations']).std()

        self.dynamics = nn.ModuleList([DynamicsModel(env, self.mu_s, self.mu_a, self.std_s, self.std_a, self.std_delta, args) for _ in range(args.num_models)])
        self.dynamics_optimizer = [Adam(self.dynamics[i].parameters(), lr=args.dynamics_lr) for i in range(args.num_models)]
        self.threshold = args.threshold
        self.halt_reward = min(dataset['rewards']) - args.halt_penalty
        self.distance = nn.PairwiseDistance()

        self.replay_buffer = ReplayBuffer()
        for idx in range(len(self.dataset['observations'])):
            state = self.dataset['observations'][idx]
            action = self.dataset['actions'][idx]
            reward = self.dataset['rewards'][idx]
            next_state = self.dataset['next_observations'][idx]
            done = self.dataset['terminals'][idx]
            self.replay_buffer.push(state, action, reward, next_state, done)

        self.logger = dict(
            epoch=[],
            total_timesteps=[],
            total_loss=[]
        )
        print('Finished initializing...')

    def train(self):
        print('#'*50)
        print('Dynamics training starts...')
        print('#'*50)
        print('\n')
        
        self.total_timesteps = 0
        for epoch in range(self.args.dynamics_epochs):
            total_loss = 0
            for i in range(self.args.num_models):
                finish = False
                for _ in range(self.args.dynamics_updates):
                    state, action, _, next_state, _ = self.replay_buffer.sample(self.args.dynamics_batch_size[i])
                    state = torch.tensor(state, dtype=torch.float)
                    action = torch.tensor(action, dtype=torch.float)
                    next_state = torch.tensor(next_state, dtype=torch.float)
                    next_state_from_dynamics = self.dynamics[i].get_next_state(state, action)

                    loss = F.mse_loss(next_state_from_dynamics * 10, next_state * 10)

                    self.dynamics_optimizer[i].zero_grad()
                    loss.backward()
                    self.dynamics_optimizer[i].step()

                    self.total_timesteps += 1

                    if self.total_timesteps >= self.args.max_timesteps:
                        finish = True
                        break

                total_loss += loss
                    
                if finish:
                    break
                    
            if epoch % self.args.logging_freq == 0:
                self.logger['epoch'].append(epoch + 1)
                self.logger['total_timesteps'].append(self.total_timesteps)
                self.logger['total_loss'].append(total_loss)
                self.logging()

            if self.args.render and epoch % self.args.render_freq == 0:
                self.env.render()
            
            if self.args.save and epoch % self.args.save_freq == 0:
                torch.save(self.dynamics.state_dict(), f'./models/dynamics_{self.args.env}.pt')
                print('Model saved!!!!!!!!!!!!')

            if finish:
                break

        # max_disc = -np.inf
        # for idx in range(len(self.dataset['observations'])):
        #     state = self.dataset['observations'][idx]
        #     action = self.dataset['actions'][idx]
        #     ns = torch.tensor([list(self.dynamics[i].get_next_state(state, action).squeeze()) for i in range(self.args.num_models)])
        #     max_disc = max(torch.max(torch.cdist(ns, ns)).item(), max_disc)
        #     if idx % 1000 == 0:
        #         print(idx)
        # self.threshold = self.args.threshold * max_disc
        # torch.save(self.threshold, f'./models/threshold_{self.args.env}.pt')
        
        print('#'*50)
        print('Dynamics training finished...')      
        print('#'*50)
        print('\n')
    
    def load(self):
        print('Model loading...')
        self.dynamics.load_state_dict(torch.load(f'./models/dynamics_{self.args.env}.pt'))
        self.threshold = torch.load(f'./models/threshold_{self.args.env}.pt').item()

    # The functions below are for P-MDP 
    def step(self, state, action):
        state = torch.tensor(state, dtype=torch.float)
        unknown = self.USAD(state, action)
        next_state = self.ensemble(state, action).squeeze()
        reward = self.get_reward(state, action, next_state, unknown)
        return next_state.cpu().numpy(), reward, unknown

    def USAD(self, state, action):
        max_disc = -np.inf
        ns = torch.tensor([list(self.dynamics[i].get_next_state(state, action).squeeze()) for i in range(self.args.num_models)])
        max_disc = max(torch.max(torch.cdist(ns, ns)).item(), max_disc)
        return max_disc > self.threshold    # True, False

    def ensemble(self, state, action):
        cat_next = torch.tensor([])
        for i in range(self.args.num_models):
            temp = self.dynamics[i].get_next_state(state, action).unsqueeze(1)
            cat_next = torch.cat([cat_next, temp], dim=1)
        next_state = torch.mean(cat_next, dim=1)
        return next_state

    def get_reward(self, state, action, next_state, unknown):
        if unknown:
            return self.halt_reward
        else:
            reward = (next_state - state) / self.env.dt + 1.
            reward -= 1e-3 * np.square(action).sum()
            return reward.cpu().numpy()[0]

    def logging(self):
        epoch = self.logger['epoch'][0]
        total_timesteps = self.logger['total_timesteps'][0]
        total_loss = sum(self.logger['total_loss']) / (len(self.logger['total_loss']) + 1e-7)
        print('#'*50)
        print(f'epoch\t\t\t|\t{epoch}')
        print(f'total_timesteps\t\t|\t{total_timesteps}')
        print(f'total_loss\t\t|\t{total_loss}')
        print('#'*50)
        print('\n')

        for k in self.logger.keys():
            self.logger[k] = []