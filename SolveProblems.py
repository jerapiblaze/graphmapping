import multiprocessing as mp
import os
import uuid

from utilities.config import ConfigParser
from utilities.multiprocessing import MultiProcessing, IterToQueue
from utilities.dir import RecurseListDir, CleanDir
from GraphMappingProblem import GraphMappingProblem, SaveProblem, LoadProblem
from GraphMappingProblem.validate import ValidateSolution


def SolveMpWorker(queue: mp.Queue, Solver, solution_setpath: str, log_setpath:str, timelimit:int):
    while queue.qsize():
        problem_path = queue.get()
        model = Solver(problem=LoadProblem(problem_path), logpath=log_setpath, timelimit=timelimit)
        solved_problem = model.Solve()
        validated_problem = ValidateSolution(solved_problem)
        savepath = os.path.join(solution_setpath, f"{validated_problem.name}.pkl.gz")
        SaveProblem(savepath, validated_problem)
    exit()

def Main(config:dict):
    print(config)
    PROBLEM_SETNAME = config["PROBLEM_SETNAME"]
    PROBLEM_SETPATH = os.path.join("./data/problems", PROBLEM_SETNAME)
    problem_set = RecurseListDir(PROBLEM_SETPATH, ["*.pkl.gz"])

    SOLVER = str(config["SOLVER"]).split("@")
    match SOLVER[0]:
        case "ILP":
            match SOLVER[1]:
                case "CBC":
                    from Solvers.ILP.cbc import Solver
                case "SCIP":
                    from Solvers.ILP.scip import Solver
                case "CPLEX":
                    from Solvers.ILP.cplex import Solver
                case "GUROBI":
                    from Solvers.ILP.gurobi import Solver
                case _:
                    raise Exception(f"[Invalid config] SOLVER=ILP_{SOLVER[1]}")
            pass
        case _:
            raise Exception(f"[Invalid config] SOLVER={SOLVER[0]}")
    
    SOLUTION_SETPATH = os.path.join("./data/solutions", f"{PROBLEM_SETNAME}@{'_'.join(SOLVER)}")
    CleanDir(SOLUTION_SETPATH)
    LOG_SETPATH = os.path.join("./data/logs", f"{PROBLEM_SETNAME}@{'_'.join(SOLVER)}")
    CleanDir(LOG_SETPATH)
    q = IterToQueue(problem_set)

    timelimit = config["TIMELIMIT"] if config["TIMELIMIT"] >= 0 else None
    MultiProcessing(SolveMpWorker, (q, Solver, SOLUTION_SETPATH, LOG_SETPATH, timelimit),4)


def MpWorker(queue: mp.Queue):
    while queue.qsize():
        item = queue.get()
        Main(item)
    exit()

if __name__=="__main__":
    mp.set_start_method("spawn")
    config_list = ConfigParser("./configs/SolveSettings/dummy.yaml")
    q = IterToQueue(config_list)
    MultiProcessing(MpWorker, (q,), 2)