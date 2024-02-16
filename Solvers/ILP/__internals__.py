import sys
import os
import pulp

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from GraphMappingProblem import GraphMappingProblem
from GraphMappingProblem.ilp import ConvertToIlp, VARIABLE_SURFIXES
from Solvers import GraphMappingSolver

class ILPSolver(GraphMappingSolver):
    def __init__(self, problem: GraphMappingProblem, logpath:str=None, timelimit:int=None, verbose:bool=False):
        self.SOLVER = pulp.LpSolver()
        self.PROBLEM = problem
        self.ILP_PROBLEM = ConvertToIlp(self.PROBLEM)
        pass

    def Solve(self) -> GraphMappingProblem:
        self.ILP_PROBLEM.solve(solver=self.SOLVER)
        obj_value = self.ILP_PROBLEM.objective.value()
        solution = {str(var): pulp.value(var) 
                    for var in self.ILP_PROBLEM.variables() 
                    if not pulp.value(var) == 0 and any(str(var).startswith(filter) for filter in VARIABLE_SURFIXES)}
        solver_runtime = self.ILP_PROBLEM.solutionTime
        solver_status = self.ILP_PROBLEM.status
        self.PROBLEM.solution = solution
        self.PROBLEM.status = solver_status
        self.PROBLEM.obj_value = obj_value
        self.PROBLEM.solution_time = solver_runtime
        return self.PROBLEM