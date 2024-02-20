import pickle
import gzip as gz
import numpy as np

from .env import *

class QLearningAgent():
    def __init__(self, state_space_size, action_space_size, alpha=0.01, gamma=0.8, epsilon=0.009):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.random.uniform(0, 1, size=(state_space_size, action_space_size))

    def choose_action(self, cr_s, trainmode:int=True):
        rand = np.random.rand()
        if rand < self.epsilon and trainmode:
            None
        else:
            return np.argmax(self.q_table[cr_s])

    def update_q_table(self, obs, action, reward, next_obs):
        best_next_action = np.argmax(self.q_table[next_obs])
        # print("curent q: ", self.q_table[cr_s, action])
        self.q_table[obs, action] += self.alpha * (reward + self.gamma * self.q_table[next_obs, best_next_action] - self.q_table[obs, action])
        # print("new q: ", self.q_table[cr_s, action] )

# Train the agent, return the trained agent and the q-value during episodes
def TrainAgent(agent:QLearningAgent, env:StaticMapping2Env, nepisode:int, verbose:bool=False) -> tuple[QLearningAgent, list[float]]:
    qvalue = list()
    for ep in range(nepisode):
        obs, info = env.reset()
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = agent.choose_action(obs)
            if not action:
                action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update_q_table(obs, action, reward, next_obs)
            obs = next_obs
        if verbose:
            print(f"ep_{ep}: {env.is_full_mapping()} {obs} {info}")
        qvalue.append((ep,np.sum(agent.q_table)))
    return agent, qvalue

def SaveAgent(path:str, agent:QLearningAgent):
    with gz.open(path, "wb") as f:
        pickle.dump(agent, f)
    return

def LoadAgent(path:str) -> QLearningAgent:
    agent = None
    with gz.open(path, "rb") as f:
        agent = pickle.load(f)
    return agent