import GraphMappingProblem
import Solvers.DeepQLearn as DQL
import gymnasium as gym

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def Main():
    problem = GraphMappingProblem.LoadProblem(".\data\problems\DUMMY\graphmapping_5425580d.pkl.gz")
    # print(problem.PHY.nodes(data = True))
    # print(problem.SFC_SET.edges(data= True))
    env = DQL.env.StaticMapping2Env(problem.PHY, problem.SFC_SET, {"node_req": "req", "link_req": "req", "node_cap": "cap", "link_cap": "cap"}, 1500, 20)
    # env = gym.make("CartPole-v1")
    obs, info = env.reset()
    # agent = DQL.agent.DeepQlearnAgent(env.obs_space_size, env.action_space_size) eps_decay = 0.9976
    agent = DQL.agent.DeepQlearnAgent(env.action_space.n, len(obs), gamma=0.8, eps_decay=0.9976, eps_start=1, eps_end=0.01, replay_buffer=10000, batch_size=128, tau=0.005, lr=1e-4, hidden_layder_dim=256, update_freq=10)
    trained_agent, reward_values = DQL.agent.TrainAgent(agent, env, 2100, verbose=True, liveview=True)
    with open("./debug_dq.csv", "wt") as f:
        f.write("ep,q\n")
        for r in reward_values:
            f.write(f"{r[0]}, {r[1]}\n")
    DQL.agent.SaveAgent("./data/__internals__/DQL/DUMMY.pkl.gz", trained_agent)
    pass


if __name__ == "__main__":
    Main()
    pass
