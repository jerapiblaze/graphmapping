import pickle
import pulp
import uuid
import networkx as nx
import gzip

class GraphMappingProblem:
    def __init__(self, phy:nx.DiGraph, sfcs:list[nx.DiGraph]) -> None:
        self.name = f"graphmapping_{uuid.uuid4().hex[:8]}"
        self.PHY = phy
        self.SFC_SET = sfcs
        self.solution = None
        self.solution_time = None
        self.obj_value = None
        self.status = None # None=Unsolved, 1=Solved, 0=SolvedNoSolution, -1=ErrorOnSolve
        self.solution_status = None # None=Unsolved, 1=OK, 0=NoSolution, -1=Invalid
    

def SaveProblem(path:str, problem:GraphMappingProblem) -> None:
    with gzip.open(path, "wb") as f:
        pickle.dump(problem, f)

def LoadProblem(path:str) -> GraphMappingProblem:
    problem = None
    with gzip.open(path, "rb") as f:
        problem = pickle.load(f)
    return problem
