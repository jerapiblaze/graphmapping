from .__internals__ import *

class Solver(ILPSolver):
    def __init__(self, problem: GraphMappingProblem, logpath: str = None, timelimit: int = None, verbose: bool = False):
        super().__init__(problem, logpath, timelimit, verbose)
        logfile = os.path.join(logpath, f"{problem.name}.log") if logpath else None
        self.SOLVER = pulp.SCIP_CMD(
            msg=verbose,
            options=["-l", logfile] if logfile else [],
            timeLimit=timelimit 
        )