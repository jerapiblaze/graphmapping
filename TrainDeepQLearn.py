import GraphMappingProblem
import Solvers.DeepQLearn as DQL
import gymnasium as gym

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def Main():
    problem = GraphMappingProblem.LoadProblem("./data/problems/DUMMY/graphmapping_1114cf92.pkl.gz")
    # print(problem.PHY.nodes(data = True))
    # print(problem.SFC_SET.edges(data= True))
    env = DQL.env3.StaticMapping2Env(problem.PHY, problem.SFC_SET, {"node_req": "req", "link_req": "req", "node_cap": "cap", "link_cap": "cap"}, 1500, 0.5)
    # env = gym.make("CartPole-v1")
    obs, info = env.reset()
    # agent = DQL.agent.DeepQlearnAgent(env.obs_space_size, env.action_space_size) eps_decay = 0.9976
    agent = DQL.agent3.DeepQlearnAgent(env.action_space.n, len(obs), 
                                       gamma=0.8, eps_decay=0.9976, eps_start=1, eps_end=0.01, 
                                       replay_buffer=10000, batch_size=128, tau=0.005, lr=0.001, hidden_layder_dim=128, 
                                       update_freq=1)
    trained_agent, reward_values = DQL.agent3.TrainAgent(agent, env, 20000, verbose=True, liveview=True)
    with open("./debug_dq3.csv", "wt") as f:
        f.write("ep,q\n")
        for r in reward_values:
            f.write(f"{r[0]}, {r[1]}\n")
    DQL.agent3.SaveAgent("./data/__internals__/DQL/DUMMY.pkl.gz", trained_agent)
    pass


if __name__ == "__main__":
    Main()
    pass
