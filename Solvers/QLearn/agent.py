import pickle
import gzip as gz
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as colors
import torch

# set up matplotlib
IS_IPYTHON = 'inline' in matplotlib.get_backend()
if IS_IPYTHON:
    from IPython import display
plt.ion()

from .env import *

class QLearningAgent():
    def __init__(self, state_space_size, action_space_size, alpha=0.01, gamma=0.8, epsilon_max= 0.9, epsilon_min=0.009, epsilon_decay=0.9876):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = self.epsilon_max
        self.q_table = np.random.uniform(0, 1, size=(state_space_size, action_space_size))
        self.episode_duration = []

    def end_episode(self, reset=False):
        if reset:
            self.epsilon = self.epsilon_max
        else:
            new_epsilon = self.epsilon * self.epsilon_decay
            new_epsilon = new_epsilon if new_epsilon > self.epsilon_min else self.epsilon_min
            self.epsilon = new_epsilon

    def choose_action(self, cr_s, trainmode:int=True):
        rand = np.random.rand()
        if rand < self.epsilon and trainmode:
            None
        else:
            return np.argmax(self.q_table[cr_s])

    def update_q_table(self, obs, action, reward, next_obs):
        # print()
        if next_obs >= self.state_space_size:
            new_q = (1-self.alpha)* self.q_table[obs, action] +self.alpha * reward
            self.q_table[obs, action] = new_q
            # self.q_table[obs, action] += (1-self.alpha)* self.q_table[obs, action] +self.alpha * reward
        else:
            best_next_action = np.argmax(self.q_table[next_obs])
            # print("current q: ", self.q_table[obs, action])
            new_q = (1-self.alpha)* self.q_table[obs, action] + self.alpha * (reward + self.gamma * self.q_table[next_obs, best_next_action] - self.q_table[obs, action])
            # self.q_table[obs, action] += (1-self.alpha)* self.q_table[obs, action] + self.alpha * (reward + self.gamma * self.q_table[next_obs, best_next_action] - self.q_table[obs, action])
            self.q_table[obs, action] = new_q
            # print("new q: ", self.q_table[obs, action])

    def plot_duration(self, show_result=False):
        plt.figure(1)
        duration_t = torch.tensor(self.episode_duration, dtype=torch.float)
        if show_result:
            plt.title("QL-Result")
        else:
            plt.clf()
            plt.title("QL-Training")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative reward")
        plt.plot(duration_t.numpy(), color='silver') # plot cumulative reward
        if len(duration_t) >= 100:
            means = duration_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), color='k') # plot average reward
        plt.pause(0.001)
        if IS_IPYTHON:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

# Train the agent, return the trained agent and the q-value during episodes
def TrainAgent(agent:QLearningAgent, env:StaticMapping2Env, nepisode:int, verbose:bool=False, liveview:bool=False) -> tuple[QLearningAgent, list[float]]:
    reward_list = []
    for ep in range(nepisode):
        obs, info = env.reset()
        terminated = False
        truncated = False
        rw_list = []
        while not terminated and not truncated:
            action = agent.choose_action(obs)
            if not action:
                action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            rw_list.append(reward)
            agent.update_q_table(obs, action, reward, next_obs)
            obs = next_obs
        if verbose:
            print(f"ep_{ep}: {env.is_full_mapping()} {obs} {info}")
            pass
        agent.end_episode()
        rw = sum(rw_list)                         # cummutative reward
        # rw = float(rw_list)/len(rw_list)    # average reward
        reward_list.append((ep, rw))
        if liveview:
            agent.episode_duration.append(rw)
            agent.plot_duration()
    return agent, reward_list

def SaveAgent(path:str, agent:QLearningAgent):
    with gz.open(path, "wb") as f:
        pickle.dump(agent, f)
    return

def LoadAgent(path:str) -> QLearningAgent:
    agent = None
    with gz.open(path, "rb") as f:
        agent = pickle.load(f)
    return agent