"""
TD3 implementation for autonomous vehicle with ROS and Gazebo
algo source: https://github.com/MrSyee/pg-is-all-you-need
"""

from envs import GazeboAutoVehicleEnv
from util import interrupt_handler
from util import ReplayBuffer, ActionNormalizer
from util import ActionNoise, NormalActionNoise, OUNoise

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

import signal
import random
from typing import List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, init_w: float = 3e-3):
        super(Actor, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, out_dim)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        action = self.out(x).tanh()

        return action


class Critic(nn.Module):
    def __init__(self, in_dim: int, init_w: float = 3e-3):
        super(Critic, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)

        return value


class TD3Agent():
    def __init__(
        self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        gamma: float = 0.99,
        tau: float = 5e-3,
        exploration_noise: float = 0.1,
        target_policy_noise: float = 0.2,
        target_policy_noise_clip: float = 0.5,
        policy_update_freq: int = 2,
        initial_random_steps: int = int(1e4),
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        wd_actor: float = 1e-2,
        model_filename: str = "td3"
    ):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.env = env
        self.memory = ReplayBuffer(obs_dim, action_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
        self.policy_update_freq = policy_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.exploration_noise = NormalActionNoise(
            0, exploration_noise, action_dim
        )
        self.target_policy_noise = NormalActionNoise(
            0, target_policy_noise, action_dim
        )
        self.target_policy_noise_clip = target_policy_noise_clip

        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target1 = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target1.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target2 = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target2.load_state_dict(self.critic2.state_dict())

        # concat critic parameters to use one optim
        critic_parameters = list(self.critic1.parameters()) + list(self.critic2.parameters())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(critic_parameters, lr=lr_critic)

        self.transition = list()
        self.total_steps = 0
        self.update_step = 0
        self.is_test = False

        self.max_score = float('-inf')
        self.model_filename = model_filename
        self.start_storing = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        if self.total_steps < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = (
                self.actor(torch.FloatTensor(state).to(self.device))[0]
                .detach()
                .cpu()
                .numpy()
            )

        if not self.is_test:
            if self.exploration_noise is not None:
                noise = self.exploration_noise.sample()
                selected_action = np.clip(
                    selected_action + noise, -1.0, 1.0
                )

            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, done, _ = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        device = self.device  # for shortening the following lines

        samples = self.memory.sample_batch()
        states = torch.FloatTensor(samples["obs"]).to(device)
        next_states = torch.FloatTensor(samples["next_obs"]).to(device)
        actions = torch.FloatTensor(samples["acts"]).to(device)
        rewards = torch.FloatTensor(samples["rews"]).to(device)
        dones = torch.FloatTensor(samples["done"]).to(device)
        masks = 1 - dones

        next_actions = self.actor_target(next_states)

        if self.target_policy_noise is not None:
            noise = torch.clamp(
                torch.FloatTensor(self.target_policy_noise.sample()).to(device),
                -self.target_policy_noise_clip,
                self.target_policy_noise_clip
            )
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)

        # min (Q_1', Q_2')
        next_values1 = self.critic_target1(next_states, next_actions)
        next_values2 = self.critic_target2(next_states, next_actions)
        next_values = torch.min(next_values1, next_values2)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_returns = rewards + self.gamma * next_values * masks
        curr_returns = curr_returns.detach()

        # critic loss
        values1 = self.critic1(states, actions)
        values2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(values1, curr_returns)
        critic2_loss = F.mse_loss(values2, curr_returns)

        # train critic
        critic_loss = critic1_loss + critic2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_steps % self.policy_update_freq == 0:
            # train actor
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # target update
            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)

        return actor_loss.data, critic_loss.data

    def train(self, num_frames: int, plotting_interval: int = 200):
        self.is_test = False

        state = self.env.reset()
        actor_losses = []
        critic_losses = []
        scores = []
        avgscores = []
        score = 0
        episode = 1
        prev_episode_steps = 0

        for self.total_steps in range(1, num_frames + 1):
            print("TOTAL STEP:", self.total_steps)
            print("EPISODE: %d - %d" % (episode, self.total_steps - prev_episode_steps))
            # print("SCORES:", scores)

            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            if done:
                state = self.env.reset()
                if self.start_storing:
                    scores.append(score)
                    # scores.append(score/(self.total_steps - prev_episode_steps))
                    avgscores.append(np.mean(scores[-50:]))
                if score > self.max_score and self.start_storing:
                    self.save(directory="./saves",
                              filename=self.model_filename+"_"+str(episode))
                    self.max_score = score

                score = 0
                episode += 1
                prev_episode_steps = self.total_steps

                if (not self.start_storing
                    and self.total_steps > self.initial_random_steps):
                    self.start_storing = True

            if len(self.memory) >= self.batch_size and self.start_storing:
                actor_loss, critic_loss = self.update_model()
                if self.total_steps % self.policy_update_freq == 0:
                    actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

        self._plot(self.total_steps, scores, avgscores,
                   actor_losses, critic_losses)

    def test(self):
        self.is_test = True

        state = self.env.reset()
        done = False
        score = 0

        frames = []
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)

        return frames

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau
        for t_param, l_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
            self.critic_target1.parameters(), self.critic1.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
            self.critic_target2.parameters(), self.critic2.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def _plot(
        self,
        frame_idx: int,
        scores: List[float],
        avgscores: List[float],
        actor_losses: List[float],
        critic_losses: List[float],
    ):

        fig, ax = plt.subplots(3, 1, figsize=(30, 15))
        ax[0].plot(scores)
        ax[0].plot(avgscores)
        ax[1].plot(actor_losses)
        ax[2].plot(critic_losses)
        plt.show()

    def save(self, directory, filename):
        print("Saving model . . .")
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))

    def load(self, directory, filename):
        print("Loading model . . .")
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))

"""## Environment
*ActionNormalizer* is an action wrapper class to normalize the action values
ranged in (-1. 1). Thanks to this class, we can make the agent simply select
action values within the zero centered range (-1, 1).
"""

class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action


signal.signal(signal.SIGINT, interrupt_handler)
env = GazeboAutoVehicleEnv(600, 800)
env = ActionNormalizer(env)


model_filename = "td3_1"

agent = TD3Agent(
    env,
    memory_size = 100000,
    batch_size = 128,
    gamma = 0.9,
    tau = 0.1,
    policy_update_freq = 2,
    initial_random_steps = 10000,
    lr_actor = 1e-4,
    lr_critic = 3e-4,
    wd_actor = 0,
    model_filename = model_filename
)


agent.train(num_frames = 50000)

# model_filename += "_7"
# agent.load(directory="./saves", filename=model_filename)
# while True:
#     agent.test()
