import os
import datetime
import multiprocessing as mp
from utilities.dir import RecurseListDir
from utilities.multiprocessing import MultiProcessing, IterToQueue

from GraphMappingProblem import GraphMappingProblem, LoadProblem

def MpWorker(queue:mp.Queue, result_file:str):
    while queue.qsize():
        solved_problem_path = queue.get()
        solved_problem = LoadProblem(solved_problem_path)
        prob_set_info = str(os.path.basename(os.path.dirname(solved_problem_path))).split("@")
        set_name = prob_set_info[0]
        solver_name = prob_set_info[1]
        problem_name = solved_problem.name
        status = solved_problem.status
        solution_status = solved_problem.solution_status
        obj_value = solved_problem.obj_value if solution_status == 1 else 0
        runtime = solved_problem.solution_time
        with open(result_file, "at") as f:
            f.write(f"{set_name},{solver_name},{problem_name},{status},{solution_status},{abs(obj_value)},{runtime}\n")
    pass

def Main():
    solved_problem_paths = RecurseListDir("./data/solutions", ["*.pkl.gz"])
    result_file = os.path.join(f"./data/results/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(result_file, "wt") as f:
        f.write(f"setname,solvername,problemname,status,solutionstatus,objvalue,runtime\n")
    q = IterToQueue(solved_problem_paths)
    MultiProcessing(MpWorker, (q, result_file), 4)

if __name__=="__main__":
    Main()
    pass