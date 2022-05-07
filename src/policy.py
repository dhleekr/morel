import numpy as np
import torch
import torch.nn.functional as F
from .networks import Actor, Critic
from torch.optim import Adam


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


class PPO:
    def __init__(self, env, dynamics, args):
        self.env = env
        self.dynamics = dynamics
        self.args = args
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.actor = Actor(env, args)
        self.critic = Critic(env, args)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.args.policy_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.args.value_lr)

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
            reward=[],
            value=[],
            rtgs=[],
            log_prob=[],
        )

    def train(self):
        print('#'*50)
        print('Policy training starts...')
        print('#'*50)
        print('\n')

        total_timesteps = 0
        for epoch in range(self.args.epochs):
            batch_states, batch_actions, batch_log_probs, batch_rtgs = self.rollout()
            print(batch_states.shape)
            with torch.no_grad():
                values = self.critic(batch_states).view(-1)
            advantage = batch_rtgs - values
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-7)

            self.logger['advantage'].append(advantage.mean())
            self.logger['rtgs'].append(batch_rtgs.mean())

            for _ in range(self.args.updates):
                total_timesteps += 1

                values = self.critic(batch_states).view(-1)
                log_probs, dist_entropy = self.actor.evaluate_action(batch_states, batch_actions)
                ratios = torch.exp(log_probs - batch_log_probs).view(-1)

                surr1 = ratios * advantage
                surr2 = torch.clamp(ratios, 1 - self.args.clip, 1 + self.args.clip) * advantage

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, batch_rtgs)
                entropy_loss = -torch.mean(dist_entropy)

                total_loss = critic_loss * self.args.value_loss_coef + actor_loss + self.args.entropy_coef * entropy_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                total_loss.backward()

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # logging 
                self.logger['total_loss'].append(total_loss)
                self.logger['actor_loss'].append(actor_loss)
                self.logger['critic_loss'].append(critic_loss)
                self.logger['entropy_loss'].append(entropy_loss)
                self.logger['ratios'].append(ratios.mean())
                self.logger['surr1'].append(surr1.mean())
                self.logger['surr2'].append(surr2.mean())
                self.logger['value'].append(values.mean())
                self.logger['log_prob'].append(log_probs.mean())
            
            if epoch % self.args.logging_freq == 0:
                self.logger['epoch'].append(epoch + 1)
                self.logger['total_timesteps'].append(total_timesteps)
                self.logging()
            
            if epoch % self.args.eval_freq == 0:
                self.evaluation()

            if self.args.save and epoch % self.args.save_freq == 0:
                torch.save(self.actor.state_dict(), f'./models/actor_{self.args.env}.pt')
                print('Model saved!!!!!!!!!!!!')

            for k in self.logger.keys():
                self.logger[k] = []

        print('#'*50)
        print('Policy training finished...')
        print('#'*50)
        print('\n')     

    def rollout(self):
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_rtgs = []

        for episode in range(self.args.num_traj):
            state = self.env.reset()
            t = 0
            done = False
            episode_reward = []

            while True:
                with torch.no_grad():
                    action, log_prob = self.actor.get_action(state)
                    action, log_prob = action.cpu().numpy(), log_prob.cpu().numpy()
                    next_state, reward, done = self.dynamics.step(state, action)

                batch_states.append(state)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                episode_reward.append(reward)

                t += 1

                if t >= self.args.max_episode_len:
                    done = True

                if done:
                    break            

                state = next_state

            batch_rewards.append(episode_reward)

        self.logger['reward'].append(sum(sum(row) / self.args.max_episode_len for row in batch_rewards))

        batch_states = torch.tensor(batch_states, dtype=torch.float)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float).view(-1, self.act_dim)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).view(-1, 1)
        with torch.no_grad():
            batch_rtgs = self.compute_rtgs(batch_rewards)
        return batch_states, batch_actions, batch_log_probs, batch_rtgs

    def compute_rtgs(self, batch_rewards):
        batch_rtgs = []

        for ep_rews in reversed(batch_rewards):
            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.args.gamma
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        batch_rtgs = (batch_rtgs - batch_rtgs.mean()) / (batch_rtgs.std() + 1e-7)
        return batch_rtgs

    def evaluation(self):
        print('\n')
        print('#'*50)
        print('Start Evaluation!!!!!!!!!!!')

        with torch.no_grad():
            avg_episode_reward = 0
            total_episode_steps = 0
            for _ in range(10):
                # self.env.seed(random.randint(0, 1000))
                state = self.env.reset()
                done = False
                episode_steps = 0
                episode_reward = 0
                for _ in range(self.args.max_episode_len):
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
            print('\n\n\n')

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
        reward = sum(self.logger['reward']) / (len(self.logger['reward']) + 1e-7)
        rtgs = sum(self.logger['rtgs']) / (len(self.logger['rtgs']) + 1e-7)
        log_prob = sum(self.logger['log_prob']) / (len(self.logger['log_prob']) + 1e-7)

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
        print(f'reward\t\t\t|\t{reward}')
        print(f'rtgs\t\t\t|\t{rtgs}')
        print(f'log_prob\t\t|\t{log_prob}')
        print('#'*50)
        print('\n')