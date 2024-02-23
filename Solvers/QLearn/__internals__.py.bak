import sys
import os
import time
from . import agent
from . import env

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from GraphMappingProblem import GraphMappingProblem
from Solvers import GraphMappingSolver

class Solver(GraphMappingSolver):
    def __init__(self, problem: GraphMappingProblem, agentpath: str, logpath: str = None, timelimit: int = None, verbose: bool = False):
        self.problem = problem
        self.agent = agent.LoadAgent(agentpath)
        self.env = env.StaticMapping2Env(problem.PHY, problem.SFC_SET, key_attrs={"node_req": "req", "link_req": "req", "node_cap": "cap", "link_cap": "cap"}, M=200, beta=5)
        self.logpath = logpath
        self.timelimit = timelimit
        self.verbose = verbose

    def Solve(self,) -> GraphMappingProblem:
        terminated = False
        truncated = False
        obs, info = self.env.reset()
        timest = time.perf_counter()
        timedr = 0
        self.problem.status = 1
        while not terminated and not truncated:
            action = self.agent.choose_action(obs, trainmode=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            obs = next_obs
            timenow = time.perf_counter()
            timedr = abs(timest-timenow)
            if timedr >= self.timelimit:
                break
        self.problem.solution_time = timedr
        self.problem.solution = self.env.render()
        self.problem.obj_value = len([x for x in self.problem.solution.keys() if str(x).__contains__("xSFC")])
        return self.problem
