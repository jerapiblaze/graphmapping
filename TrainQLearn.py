import GraphMappingProblem
import Solvers.QLearn as QLearn
import gymnasium as gym

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def Main():
    problem = GraphMappingProblem.LoadProblem("./data/problems/DUMMY/graphmapping_1114cf92.pkl.gz")
    # print(problem.PHY.nodes(data = True))
    # print(problem.SFC_SET.edges(data= True))
    env = QLearn.env.StaticMapping2Env(problem.PHY, problem.SFC_SET, {"node_req": "req", "link_req": "req", "node_cap": "cap", "link_cap": "cap"}, 1500, 20)
    
    n_episodes = 5000
    alpha = 0.009
    gamma = 0.8
    epsilon_max, epsilon_min = 1, 0.01 
    epsilon_decay = (epsilon_max-epsilon_min)/n_episodes #0.9954
    
    agent = QLearn.agent.QLearningAgent(env.obs_space_size, env.action_space_size, 
                                        alpha=alpha, gamma=gamma, epsilon_max=epsilon_max, 
                                        epsilon_min=epsilon_min, epsilon_decay=epsilon_decay)
    trained_agent, reward_values = QLearn.agent.TrainAgent(agent, env, n_episodes, verbose=True, liveview=True)
    with open("./debug_q.csv", "wt") as f:
        f.write("ep,q\n")
        for r in reward_values:
            f.write(f"{r[0]}, {r[1]}\n")
    QLearn.agent.SaveAgent("./data/__internals__/QLearn/DUMMY.pkl.gz", trained_agent)
    pass

if __name__=="__main__":
    Main()
    pass