import GraphMappingProblem
import Solvers.DeepQLearn as DQL
import gymnasium as gym
from utilities.config import ConfigParser
from utilities.dir import RecurseListDir, CleanDir

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def Main(config):
    print(config)
    if config["DELETE_OLD_DATA"]:
        CleanDir("./data/__internals__/QL")
    problemset_name = config["PROBLEM_SETNAME"]
    save_reward = config["SAVE_REWARDS"]
    liveview, verbose = config["LIVEVIEW"], config["VERBOSE"]
    problem_dir = os.path.join(f"./data/problems/{problemset_name}")
    n_episode = config["N_EPISODES"]
    gamma = config["GAMMA"]
    epsilon_start, epsilon_end, epsilon_decay = config["EPSILON_START"], config["EPSILON_END"], config["EPSILON_DECAY"]
    tau, lr, update_freq, hidden_layer_dim = config["TAU"], config["LR"], config["UPDATE_FREQ"], config["HIDDEN_LAYER_DIM"]
    batch_size, replay_buffer_size = config["BATCH_SIZE"], config["REPLAY_BUFFER_SIZE"]
    big_m, beta = config["BIG_M"], config["BETA"]
    problem_path_list = RecurseListDir(problem_dir, ["*.pkl.gz"])
    for problem_path in problem_path_list:
        problem = GraphMappingProblem.LoadProblem(problem_path)
        print(problem.name)
        env = DQL.env3.StaticMapping2Env(problem.PHY, problem.SFC_SET, {"node_req": "req", "link_req": "req", "node_cap": "cap", "link_cap": "cap"}, big_m, beta)
        obs, info = env.reset()
        agent = DQL.agent3.DeepQlearnAgent(env.action_space.n, len(obs), 
                                            gamma=gamma, eps_decay=epsilon_decay, eps_start=epsilon_start, eps_end=epsilon_end, 
                                            replay_buffer=replay_buffer_size, batch_size=batch_size, tau=tau, lr=lr, hidden_layder_dim=hidden_layer_dim, 
                                            update_freq=update_freq)
        trained_agent, reward_values = DQL.agent3.TrainAgent(agent, env, n_episode, verbose, liveview)
        model_save_path = os.path.join("./data/__internals__/DQL", f"{problem.name}.pkl.gz")
        DQL.agent3.SaveAgent(model_save_path, trained_agent)
        if not save_reward:
            continue
        rewards_save_path = os.path.join("./data/__internals__/DQL", f"{problem.name}_rewards.csv")
        with open(rewards_save_path, "wt") as f:
            f.write("ep, reward\n")
            for r in reward_values:
                f.write(f"{r[0]}, {r[1]}\n")

if __name__=="__main__":
    config_list = ConfigParser("./configs/DQLSettings/dummy.yaml")
    for config in config_list:
        Main(config)