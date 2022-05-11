import argparse

import gym
import d4rl
import torch
from torch.utils.data import DataLoader
from src.dynamics import Dynamics
from src.policy import PPO
from src.dataset import OfflineDataset

parser = argparse.ArgumentParser(description="MOReL training.")
parser.add_argument('--gamma', dest='gamma', default=0.99, type=float)
parser.add_argument('--gae_lambda', dest='gae_lambda', default=0.95, type=float)
parser.add_argument('--clip', dest='clip', default=0.1, type=float)
parser.add_argument('--value_loss_coef', dest='value_loss_coef', default=0.5, type=float)
parser.add_argument('--entropy_coef', dest='entropy_coef', default=0.01, type=float)
parser.add_argument('--num_models', dest='num_models', default=4, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=1024, type=int)
parser.add_argument('--dynamics_batch_size', dest='dynamics_batch_size', default=256, type=int)
parser.add_argument('--dynamics_hidden_dim', nargs='+', dest='dynamics_hidden_dim', default=[512, 512], type=int)
parser.add_argument('--policy_hidden_dim', nargs='+', dest='policy_hidden_dim', default=[256, 256], type=int)
parser.add_argument('--value_hidden_dim', nargs='+', dest='value_hidden_dim', default=[256, 256], type=int)
parser.add_argument('--dynamics_lr', dest='dynamics_lr', default=5e-4, type=float)
parser.add_argument('--policy_lr', dest='policy_lr', default=3e-4, type=float)
parser.add_argument('--value_lr', dest='value_lr', default=3e-4, type=float)
parser.add_argument('--halt_penalty', dest='halt_penalty', default=50, type=int)
parser.add_argument('--buffer_size', dest='buffer_size', default=int(2e6), type=int)
parser.add_argument('--threshold', dest='threshold', default=5., type=float)

parser.add_argument('--max_timesteps', dest='max_timesteps', default=int(3e7), type=int)
parser.add_argument('--dynamics_epochs', dest='dynamics_epochs', default=100, type=int)
parser.add_argument('--epochs', dest='epochs', default=100000, type=int)
parser.add_argument('--bc_epochs', dest='bc_epochs', default=5, type=int)
parser.add_argument('--num_traj', dest='num_traj', default=2000, type=int)
parser.add_argument('--max_episode_len', dest='max_episode_len', default=500, type=int)
parser.add_argument('--updates', dest='updates', default=10, type=int)
parser.add_argument('--env', dest='env', default='hopper-expert-v2', type=str)

parser.add_argument('--mode', dest='mode', default='train', type=str)
parser.add_argument('--dynamics', dest='dynamics', default='train', type=str)
parser.add_argument('--save', dest='save', default=True, type=bool)
parser.add_argument('--render', dest='render', default=False, type=bool)

parser.add_argument('--render_freq', dest='render_freq', default=1, type=int)
parser.add_argument('--logging_freq', dest='logging_freq', default=1, type=int)
parser.add_argument('--save_freq', dest='save_freq', default=1, type=int)
parser.add_argument('--eval_freq', dest='eval_freq', default=1, type=int)

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

env = gym.make(args.env)
dataset = OfflineDataset(env)
dataloader = DataLoader(dataset, batch_size=args.dynamics_batch_size, shuffle=True)
dynamics = Dynamics(env, dataset, args)
agent = PPO(env, dynamics, args)

if args.mode == 'train':
    # 1. Learning dynamics
    if args.dynamics == 'train':
        dynamics.train(dataloader)
    elif args.dynamics == 'load':
        dynamics.load()
        agent.actor.load_state_dict(torch.load(f'./models/actor_{args.env}.pt', map_location=device))
    # 2. Learning policy using learned dynamics (P-MDP)
    agent.train()
    
elif args.mode == 'test':
    agent.test()

else:
    dynamics.get_max_disc(dataset)
