import GraphMappingProblem
import Solvers.QLearn as QLearn
import gymnasium as gym

def Main():
    problem = GraphMappingProblem.LoadProblem("./data/problems/DUMMY/graphmapping_10980c38.pkl.gz")
    env = QLearn.env.StaticMapping2Env(problem.PHY, problem.SFC_SET, {"node_req": "req", "link_req": "req", "node_cap": "cap", "link_cap": "cap"}, 500, 5)
    agent = QLearn.agent.QLearningAgent(env.obs_space_size, env.action_space_size, alpha=0.001, gamma=0.01, epsilon=0.009)
    trained_agent, q_values = QLearn.agent.TrainAgent(agent, env, 1000, True)
    with open("./debug_q.csv", "wt") as f:
        f.write("ep,q\n")
        for q in q_values:
            f.write(f"{q[0]},{q[1]}\n")
    QLearn.agent.SaveAgent("./data/__internals__/QLearn/DUMMY.pkl.gz", trained_agent)
    pass

if __name__=="__main__":
    Main()
    pass