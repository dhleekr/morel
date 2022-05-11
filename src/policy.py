from tqdm import tqdm
import collections
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from src.networks import Actor, Critic


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


class PPO:
    def __init__(self, env, dynamics, args):
        self.env = env
        self.dynamics = dynamics
        self.args = args
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.actor = Actor(env, args).to(device)
        self.critic = Critic(env, args).to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.args.policy_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.args.value_lr)

        self.BC_dataloader = DataLoader(BC_Dataset(env), batch_size=args.batch_size, shuffle=True)

        self.logger = dict(
            epoch=[],
            total_timesteps=[],
            total_loss=[],
            actor_loss=[],
            critic_loss=[],
            entropy_loss=[],
            advantage=[],
            ratios=[],
            surr1=[],
            surr2=[],
            value=[],
            returns=[],
            log_prob=[],
        )

        self.writer = SummaryWriter(f"./results/{self.args.env}")

    def train(self):
        print('\n')
        print('#'*50)
        print('Policy Training Starts...')
        print('#'*50)
        print('\n')

        self.total_timesteps = 0
        for epoch in range(self.args.epochs):
            if self.args.bc_epochs > epoch:
                self.behavior_cloning()
            else:
                batch_states, batch_actions, batch_returns, batch_log_probs = self.rollout()
                
                self.logger['returns'].append(batch_returns.mean())

                for _ in tqdm(range(self.args.updates)):
                    self.total_timesteps += 1
                    values = self.critic(batch_states).squeeze()
                    log_probs, dist_entropy = self.actor.evaluate_action(batch_states, batch_actions)
                    log_probs = log_probs.squeeze()

                    advantage = batch_returns - values
                    # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-7)

                    ratios = torch.exp(log_probs - batch_log_probs)

                    surr1 = ratios * advantage
                    surr2 = torch.clamp(ratios, 1 - self.args.clip, 1 + self.args.clip) * advantage
                    
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = F.mse_loss(values, batch_returns)
                    entropy_loss = -torch.mean(dist_entropy)     

                    total_loss = critic_loss * self.args.value_loss_coef + actor_loss + self.args.entropy_coef * entropy_loss

                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()

                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

                    # logging 
                    self.logger['total_loss'].append(total_loss)
                    self.logger['actor_loss'].append(actor_loss)
                    self.logger['critic_loss'].append(critic_loss)
                    self.logger['entropy_loss'].append(entropy_loss)
                    self.logger['advantage'].append(advantage.mean())
                    self.logger['ratios'].append(ratios.mean())
                    self.logger['surr1'].append(surr1.mean())
                    self.logger['surr2'].append(surr2.mean())
                    self.logger['value'].append(values.mean())
                    self.logger['log_prob'].append(log_probs.mean())

            if epoch % self.args.logging_freq == 0:
                self.logger['epoch'].append(epoch + 1)
                self.logger['total_timesteps'].append(self.total_timesteps)
                self.logging()
            
            if epoch % self.args.eval_freq == 0:
                self.evaluation()

            if self.args.save and epoch % self.args.save_freq == 0:
                torch.save(self.actor.state_dict(), f'./models/actor_{self.args.env}.pt')
                print('Model saved!!!!!!!!!!!!')

        self.writer.flush()
        self.writer.close()

        print('#'*50)
        print('Policy Training Finished...')
        print('#'*50)
        print('\n')     

    def rollout(self):
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_dones = []
        batch_values = []

        with torch.no_grad():
            t = 0
            state = self.dynamics.reset()
            done = False
            for _ in range(self.args.num_traj):     
                action, log_prob = self.actor.get_action(state)
                value = self.critic(state)
                action = torch.squeeze(action)
                log_prob = torch.squeeze(log_prob)
                value = torch.squeeze(value)

                batch_states.append(state)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                batch_values.append(value)

                next_state, reward, done, _ = self.dynamics.step(state, action)

                batch_rewards.append(reward)

                t += 1
                if t >= self.args.max_episode_len:
                    done = True

                batch_dones.append(done)

                if done:
                    state = self.dynamics.reset()
                    done = False
                    t = 0
                
                state = next_state

            batch_states = torch.stack(batch_states)
            batch_actions = torch.stack(batch_actions)
            batch_log_probs = torch.stack(batch_log_probs)
            batch_rewards = torch.stack(batch_rewards)
            batch_values = torch.stack(batch_values)
            batch_dones = np.asarray(batch_dones, dtype=np.bool_)

            print(collections.Counter(batch_dones))

            last_value = self.critic(state)

            batch_advantages = torch.zeros_like(batch_rewards).float().to(device)
            last_gae_lam = torch.Tensor([0.0]).float().to(device)

            for t in reversed(range(self.args.num_traj)):
                if t == self.args.num_traj - 1:
                    next_non_terminal = 1.0 - done
                    next_values = last_value
                else:
                    next_non_terminal = 1.0 - batch_dones[t + 1]
                    next_values = batch_values[t + 1]

                delta = batch_rewards[t] + self.args.gamma * next_values * next_non_terminal - batch_values[t]
                batch_advantages[t] = last_gae_lam = delta + self.args.gamma * self.args.gae_lambda * next_non_terminal * last_gae_lam
            
            batch_returns = batch_advantages + batch_values

        return batch_states, batch_actions, batch_returns, batch_log_probs

    def behavior_cloning(self):
        for i, batch in enumerate(tqdm(self.BC_dataloader)):
            self.total_timesteps += 1
            feed, target = batch
            log_prob, _ = self.actor.evaluate_action(feed, target)
            actor_loss = -log_prob.sum()

            self.actor_optimizer.zero_grad()
                    
            actor_loss.backward()

            self.actor_optimizer.step()

            self.logger['actor_loss'].append(actor_loss)
 
    def evaluation(self):
        print('#'*50)
        print('Evaluation!!!!!!!!!!!')
        self.actor = self.actor.to("cpu")

        with torch.no_grad():
            avg_episode_reward = 0
            total_episode_steps = 0
            for _ in range(10):
                # self.env.seed(random.randint(0, 1000))
                state = self.env.reset()
                done = False
                episode_steps = 0
                episode_reward = 0
                while True:
                    if self.args.render:
                        self.env.render()
                    action, _ = self.actor.get_action(state)
                    action = action.cpu().numpy().squeeze()
                    next_state, reward, done, _ = self.env.step(action)
                    
                    episode_reward += reward
                    episode_steps += 1
                    state = next_state

                    if done:
                        break

                avg_episode_reward += episode_reward
                total_episode_steps += episode_steps
            
            avg_episode_reward /= 10
            avg_episode_steps = total_episode_steps / 10

            print(f"Average epdisode reward : {avg_episode_reward}")
            print(f"Average epdisode steps : {avg_episode_steps}")
            print('#'*50)
            print('\n\n')

            self.writer.add_scalar("Average return", avg_episode_reward, self.total_timesteps)
            self.actor = self.actor.to(device)
            

    def logging(self):
        epoch = self.logger['epoch'][0]
        total_timesteps = self.logger['total_timesteps'][0]
        total_loss = sum(self.logger['total_loss']) / (len(self.logger['total_loss']) + 1e-7)
        actor_loss = sum(self.logger['actor_loss']) / (len(self.logger['actor_loss']) + 1e-7)
        critic_loss = sum(self.logger['critic_loss']) / (len(self.logger['critic_loss']) + 1e-7)
        entropy_loss = sum(self.logger['entropy_loss']) / (len(self.logger['entropy_loss']) + 1e-7)
        advantage = sum(self.logger['advantage']) / (len(self.logger['advantage']) + 1e-7)
        ratios = sum(self.logger['ratios']) / (len(self.logger['ratios']) + 1e-7)
        surr1 = sum(self.logger['surr1']) / (len(self.logger['surr1']) + 1e-7)
        surr2 = sum(self.logger['surr2']) / (len(self.logger['surr2']) + 1e-7)
        value = sum(self.logger['value']) / (len(self.logger['value']) + 1e-7)
        returns = sum(self.logger['returns']) / (len(self.logger['returns']) + 1e-7)
        log_prob = sum(self.logger['log_prob']) / (len(self.logger['log_prob']) + 1e-7)

        print('\n')     
        print('#'*50)
        print(f'epoch\t\t\t|\t{epoch}')
        print(f'total_timesteps\t\t|\t{total_timesteps}')
        print(f'total_loss\t\t|\t{total_loss}')
        print(f'actor_loss\t\t|\t{actor_loss}')
        print(f'critic_loss\t\t|\t{critic_loss}')
        print(f'entropy_loss\t\t|\t{entropy_loss}')
        print(f'advantage\t\t|\t{advantage}')
        print(f'ratios\t\t\t|\t{ratios}')
        print(f'surr1\t\t\t|\t{surr1}')
        print(f'surr2\t\t\t|\t{surr2}')
        print(f'value\t\t\t|\t{value}')
        print(f'returns\t\t\t|\t{returns}')
        print(f'log_prob\t\t|\t{log_prob}')
        print('#'*50)
        print('\n')

        for k in self.logger.keys():
            self.logger[k] = []

    def test(self):
        import random     
        self.actor.load_state_dict(torch.load(f"./models/actor_{self.args.env}.pt", map_location=device))

        for _ in range(15):
            done = False
            self.env.seed(random.randint(0, 1000))
            state = self.env.reset()
            episode_steps = 0
            episode_reward = 0

            while True:
                self.env.render()
                action, _ = self.actor.get_action(state)
                action = action.cpu().numpy().squeeze()
                # action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_steps += 1
                state = next_state
                    
                if done:
                    break

            print('#'*50)
            print(f"Epdisode steps : {episode_steps}")
            print(f"Epdisode reward : {episode_reward}")
            print('#'*50)
            print('\n')


class BC_Dataset(Dataset):
    def __init__(self, env):
        dataset = env.get_dataset()
        self.observation = dataset["observations"]
        self.action = dataset["actions"]

    def __getitem__(self, idx):
        feed = torch.FloatTensor(self.observation[idx]).to(device)
        target = torch.FloatTensor(self.action[idx]).to(device)
        return feed, target

    def __len__(self):
        return len(self.observation)