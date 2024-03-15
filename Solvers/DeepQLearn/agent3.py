import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as colors
from collections import namedtuple, deque
from itertools import count
import gzip as gz
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .env3 import StaticMapping2Env

# Delect device for trainning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# REFERENCE: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# set up matplotlib
IS_IPYTHON = 'inline' in matplotlib.get_backend()
if IS_IPYTHON:
    from IPython import display
plt.ion()


# Replay memory

TRANSITION = namedtuple('TRANSITION', ('obs', 'action', 'next_obs', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(TRANSITION(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Deep Q-Learning neural network

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_dim):
        super(DQN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Linear(n_observations, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer22 = nn.Linear(hidden_dim, hidden_dim)
        self.layer222 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer22(x))
        x = F.relu(self.layer222(x))
        return self.layer3(x)

class DeepQlearnAgent:
    def __init__(self, action_space_size:int, obs_space_size:int, batch_size:int=128, hidden_layder_dim:int=128, replay_buffer:int=512, update_freq:int=1, gamma:float=0.99, eps_start:float=0.9, eps_end:float=0.05, eps_decay:float=1000, tau:float=0.005, lr:float=1e-4):
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eps = eps_start
        self.tau = tau
        self.lr = lr
        self.hidden_layer_dim = hidden_layder_dim
        self.update_freq = update_freq

        self.n_actions = action_space_size
        self.n_obs = obs_space_size
        
        self.policy_net = DQN(n_observations=self.n_obs, n_actions=self.n_actions, hidden_dim=self.hidden_layer_dim).to(DEVICE)
        self.target_net = DQN(n_observations=self.n_obs, n_actions=self.n_actions, hidden_dim=self.hidden_layer_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(replay_buffer)

        self.episode_duration = []

    def end_episode(self, reset=False):
        if reset:
            self.eps = self.eps_start
        else:
            new_eps = self.eps * self.eps_decay
            # new_eps = self.eps * math.exp(-1*self.eps_decay)
            # new_eps = self.eps - self.eps_decay
            new_eps = new_eps if new_eps > self.eps_end else self.eps_end
            self.eps = new_eps

    def select_action(self, obs, trainmode=True):
        if not trainmode:
            obs = torch.tensor(obs, device=DEVICE, dtype=torch.float32).unsqueeze(0)
            return self.target_net(obs).max(1).indices.view(1, 1)
        sample = np.random.rand()
        if sample > self.eps:
            with torch.no_grad():
                return self.policy_net(obs).max(1).indices.view(1, 1)
        else:
            None
        
    def plot_duration(self, show_result=False):
        plt.figure(1)
        duration_t = torch.tensor(self.episode_duration, dtype=torch.float)
        if show_result:
            plt.title("DQL-Result")
        else:
            plt.clf()
            plt.title("DQL-Training")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative reward")
        plt.plot(duration_t.numpy(), color='silver')
        if len(duration_t) >= 100:
            means = duration_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), color='darkgoldenrod') # plot average reward
        plt.pause(0.001)
        if IS_IPYTHON:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

def OptimizeAgent2(agent:DeepQlearnAgent) -> tuple[DeepQlearnAgent, float]:
    if (len(agent.memory) < agent.batch_size):
        return agent, None
    transisions = agent.memory.sample(agent.batch_size)
    loss_total = []
    for transision in transisions:
        transision = TRANSITION(*transision)
        if transision.next_obs is None:
            continue
        obs = transision.obs
        next_obs = transision.next_obs
        action = transision.action.item()
        reward = transision.reward.item()
        q_current = torch.flatten(agent.policy_net(obs))[action] # Q(s, a)
        with torch.no_grad():
            q_next = torch.flatten(agent.target_net(next_obs)).max()
        expected_q = q_next * agent.gamma + reward # Q'(s', a)
        criterition = nn.SmoothL1Loss()
        loss = criterition(q_current, expected_q) # delta = Q(s, a) - Q()
        agent.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(agent.policy_net.parameters(), 1024)
        agent.optimizer.step()
        loss_total.append(loss.item())
    loss_avg = np.average(loss_total)
    return (agent, loss_avg)

def OptimizeAgent(agent:DeepQlearnAgent) -> DeepQlearnAgent:
    if (len(agent.memory) < agent.batch_size):
        return agent, None
    transisions = agent.memory.sample(agent.batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = TRANSITION(*zip(*transisions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_obs)), device=DEVICE, dtype=torch.bool)
    non_final_next_obs = torch.cat([s for s in batch.next_obs if s is not None])
    obs_batch = torch.cat(batch.obs)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    obs_action_values = agent.policy_net(obs_batch).gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_obs_values = torch.zeros(agent.batch_size, device=DEVICE)
    with torch.no_grad():
        next_obs_values[non_final_mask] = agent.target_net(non_final_next_obs).max(1).values
    # Compute the expected Q values
    expected_obs_action_values = (next_obs_values * agent.gamma) + reward_batch
    # expected_obs_action_values = reward_batch * agent.gamma * next_obs_values
    # Compute Huber loss
    # criterion = nn.HuberLoss()
    criterion = nn.SmoothL1Loss()
    # MSE Loss
    # criterion = nn.MSELoss()
    loss = criterion(obs_action_values, expected_obs_action_values.unsqueeze(1))
    # loss = criterion(expected_obs_action_values.unsqueeze(1), obs_action_values)
    # Optimize the model
    agent.optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(agent.policy_net.parameters(), 1024)
    agent.optimizer.step()
    return agent, None

def TrainAgent(agent:DeepQlearnAgent, env: StaticMapping2Env, nepisode:int, verbose:bool=False, liveview:bool=False) -> DeepQlearnAgent:
    if verbose:
        print(f"Training on: {DEVICE}")
    reward_list = []
    for eps in range(nepisode):
        obs, info = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        rw_list = []
        for t in count():
            action = agent.select_action(obs)
            if not action:
                action = torch.tensor([[env.action_space.sample()]], device=DEVICE, dtype=torch.long)
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            rw_list.append(reward)
            reward = torch.tensor([reward], device=DEVICE)
            done = terminated or truncated
            if done:
                next_obs = None
            else:
                # next_obs = obs.clone().detach().unsqueeze(0)
                next_obs =torch.tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            agent.memory.push(obs, action, next_obs, reward)
            obs = next_obs
            if ((eps+1) % agent.update_freq == 0):
                # Perform one step of the optimization (on the policy network)
                agent, train_loss = OptimizeAgent(agent)
                # Soft update of the target network's weights every agent.update_freq
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = agent.target_net.state_dict()
                policy_net_state_dict = agent.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * agent.tau + target_net_state_dict[key] * (1 - agent.tau)
                agent.target_net.load_state_dict(target_net_state_dict)
            if done:
                agent.end_episode()
                # rw = float(sum(rw_list))/len(rw_list) # average reward
                rw = sum(rw_list)                # cumulative reward
                reward_list.append((eps,rw))
                if liveview:
                    agent.episode_duration.append(rw)
                    agent.plot_duration()
                if verbose:
                    print(f"{eps}/{nepisode} @{t} {env.vnf_order_index_current} {env.is_full_mapping()} {agent.eps} {info} {train_loss}")
                    # print(f"{eps}/{nepisode} {rw} {agent.eps} {info}")
                break
    if verbose:
        print("Completed.")
        agent.plot_duration(verbose)
        plt.ioff()
        plt.show()
    return agent, reward_list

def SaveAgent(path: str, agent: DeepQlearnAgent):
    with gz.open(path, "wb") as f:
        pickle.dump(agent, f)
    return


def LoadAgent(path: str) -> DeepQlearnAgent:
    agent = None
    with gz.open(path, "rb") as f:
        agent = pickle.load(f)
    return agent
