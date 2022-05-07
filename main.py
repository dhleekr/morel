import argparse
import gym
import d4rl
from src.dynamics import Dynamics
from src.policy import PPO

parser = argparse.ArgumentParser(description="SAC training.")
parser.add_argument('--gamma', dest='gamma', default=0.99, type=float)
parser.add_argument('--clip', dest='clip', default=0.1, type=float)
parser.add_argument('--value_loss_coef', dest='value_loss_coef', default=0.5, type=float)
parser.add_argument('--entropy_coef', dest='entropy_coef', default=0.01, type=float)
parser.add_argument('--num_models', dest='num_models', default=4, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=256, type=int)
parser.add_argument('--dynamics_batch_size', nargs='+', dest='dynamics_batch_size', default=[64, 128, 256, 512], type=int)
parser.add_argument('--dynamics_hidden_dim', nargs='+', dest='dynamics_hidden_dim', default=[512, 512], type=int)
parser.add_argument('--policy_hidden_dim', nargs='+', dest='policy_hidden_dim', default=[32, 32], type=int)
parser.add_argument('--value_hidden_dim', nargs='+', dest='value_hidden_dim', default=[128, 128], type=int)
parser.add_argument('--dynamics_lr', dest='dynamics_lr', default=5e-4, type=float)
parser.add_argument('--policy_lr', dest='policy_lr', default=3e-4, type=float)
parser.add_argument('--value_lr', dest='value_lr', default=3e-4, type=float)
parser.add_argument('--halt_penalty', dest='halt_penalty', default=50, type=int)
parser.add_argument('--buffer_size', dest='buffer_size', default=int(2e6), type=int)
parser.add_argument('--threshold', dest='threshold', default=1., type=float)

parser.add_argument('--max_timesteps', dest='max_timesteps', default=int(3e7), type=int)
parser.add_argument('--dynamics_epochs', dest='dynamics_epochs', default=10000, type=int)
parser.add_argument('--epochs', dest='epochs', default=1000, type=int)
parser.add_argument('--num_traj', dest='num_traj', default=256, type=int)
parser.add_argument('--max_episode_len', dest='max_episode_len', default=500, type=int)
parser.add_argument('--dynamics_updates', dest='dynamics_updates', default=10, type=int)
parser.add_argument('--updates', dest='updates', default=1000, type=int)
parser.add_argument('--env', dest='env', default='hopper-expert-v2', type=str)

parser.add_argument('--mode', dest='mode', default='train', type=str)
parser.add_argument('--dynamics', dest='dynamics', default='train', type=str)
parser.add_argument('--save', dest='save', default=False, type=bool)
parser.add_argument('--render', dest='render', default=False, type=bool)

parser.add_argument('--render_freq', dest='render_freq', default=1, type=int)
parser.add_argument('--logging_freq', dest='logging_freq', default=1, type=int)
parser.add_argument('--save_freq', dest='save_freq', default=1, type=int)
parser.add_argument('--eval_freq', dest='eval_freq', default=1, type=int)

args = parser.parse_args()

assert args.num_models == len(args.dynamics_batch_size), 'The number of models and the number of batch sizes are different'

env = gym.make(args.env)
dataset = d4rl.qlearning_dataset(env)
dynamics = Dynamics(env, dataset, args)
agent = PPO(env, dynamics, args)

# 1. Learning dynamics
if args.dynamics == 'train':
    dynamics.train()
    dynamics.load()
elif args.dynamics == 'load':
    dynamics.load()

# state = dataset['observations'][0]
# action = dataset['actions'][0]
# next_state = dataset['next_observations'][0]

# import torch
# with torch.no_grad():   
#     state = torch.tensor(state)
#     action = torch.tensor(action)
#     predict, _, _= dynamics.step(state, action)
#     print(next_state)
#     print(predict)
#     print(next_state - predict)
# 2. Learning policy using learned dynamics (P-MDP)
agent.train()