import networkx as nx
import matplotlib.pyplot as plt

import GraphMappingProblem

# PHY = ProblemGenerate.phy.randgraph.RandomGraphGenerator(100,100,10,0.0).Generate()
PHY = GraphMappingProblem.phy.fromgml.FromGmlGraphGenerator("./data/__internals__/SndLib/pioro40.gml", 100,100).Generate()

SFC = GraphMappingProblem.sfc.linear.LinearSfcGraphGenerator(3,5,5).Generate()

SFC_SET = [
    GraphMappingProblem.sfc.linear.LinearSfcGraphGenerator(5, [5,10], [5,10]).Generate()
    for i in range(10)
]

# plt.figure()
# nx.draw_kamada_kawai(PHY)
# plt.figure()
# nx.draw_kamada_kawai(SFC)

problem = GraphMappingProblem.GraphMappingProblem(PHY, SFC_SET)

from Solvers.ILP.scip import Solver

model = Solver(problem, verbose=False)

problem = model.Solve()

problem = GraphMappingProblem.validate.ValidateSolution(problem, debug=False)

print(problem.solution, problem.status, problem.solution_status, problem.solution_time, problem.obj_value)

GraphMappingProblem.SaveProblem("./data/temp/random.pkl.gz", problem)

prob2 = GraphMappingProblem.LoadProblem("./data/temp/random.pkl.gz")

print(prob2.solution, prob2.status, prob2.solution_status, prob2.solution_time, prob2.obj_value)
