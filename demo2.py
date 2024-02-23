import GraphMappingProblem

prob = GraphMappingProblem.LoadProblem("./data/solutions/DUMMY@ILP_SCIP/graphmapping_10980c38.pkl.gz")
# prob = GraphMappingProblem.LoadProblem("./data/solutions/DUMMY@ILP_SCIP/graphmapping_46bf0ed1.pkl.gz")

prob = GraphMappingProblem.validate.ValidateSolution(prob, debug=True)

print(prob.solution)

print(prob.PHY.name)
print(prob.PHY.nodes(data=True))
print(prob.PHY.edges(data=True))

for sfc in prob.SFC_SET:
    print(sfc.name)
    print(sfc.nodes(data=True))
    print(sfc.edges(data=True))