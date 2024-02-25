import GraphMappingProblem
import Solvers.QLearn as QLearn
import gymnasium as gym

def Main():
    problem = GraphMappingProblem.LoadProblem(".\data\problems\DUMMY\graphmapping_08842ee0.pkl.gz")
    # print(problem.PHY.nodes(data = True))
    # print(problem.SFC_SET.edges(data= True))
    env = QLearn.env.StaticMapping2Env(problem.PHY, problem.SFC_SET, {"node_req": "req", "link_req": "req", "node_cap": "cap", "link_cap": "cap"}, 1500, 20)
    
    agent = QLearn.agent.QLearningAgent(env.obs_space_size, env.action_space_size, alpha=0.009, gamma=0.8, epsilon=0.01)
    trained_agent, reward_values = QLearn.agent.TrainAgent(agent, env, 1000, True)
    with open("./debug_q.csv", "wt") as f:
        f.write("ep,q\n")
        for r in reward_values:
            f.write(f"{r[0]}, {r[1]}\n")
    QLearn.agent.SaveAgent("./data/__internals__/QLearn/DUMMY.pkl.gz", trained_agent)
    pass

if __name__=="__main__":
    Main()
    pass