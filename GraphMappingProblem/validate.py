from .__internals__ import *

from pulp import *
import networkx as nx


def ValidateSolution(prob: GraphMappingProblem, debug:bool=False) -> GraphMappingProblem:
    if (prob.status == None):
        prob.solution_status = None
        return prob
    if (prob.status == 0 or prob.status == -1 or not prob.solution):
        prob.solution_status = 0
        return prob
    
    solution_data = prob.solution
    G = prob.PHY
    SFCs = prob.SFC_SET

    # VALIDATE

    # Build Node Placement List
    phiNode_S = list()
    for sfc in SFCs:
        phiNode_S.append(
            LpVariable.dicts(
                name=f"xNode_{SFCs.index(sfc)}",
                indices=(sfc.nodes, G.nodes),
                cat="Binary"
            )
        )

    # Build Link Placement List
    phiLink_S = list()
    for sfc in SFCs:
        phiLink_S.append(
            LpVariable.dicts(
                name=f"xEdge_{SFCs.index(sfc)}",
                indices=(sfc.edges, G.edges),
                cat="Binary"
            )
        )

    phiSFC = LpVariable.dicts(
        name="xSFC",
        indices=(range(len(SFCs))),
        cat="Binary"
    )

    # C1: Node Capacity
    for node in G.nodes:
        if not (
            sum(
                sum(
                    get_solution_value(solution_data, phiNode_S[SFCs.index(sfc)][node_S][node].name) * nx.get_node_attributes(sfc, "req")[node_S]
                    for node_S in sfc.nodes
                )
                for sfc in SFCs
            )
                <= nx.get_node_attributes(G, "cap")[node]
        ):
            prob.solution_status = "c1" if debug else -1
            return prob

    # C2: Edge Capacity
    for edge in G.edges:
        if not (
            sum(
                sum(
                    get_solution_value(solution_data, phiLink_S[SFCs.index(sfc)][link_S][edge].name) * nx.get_edge_attributes(sfc, "req")[link_S]
                    for link_S in sfc.edges
                )
                for sfc in SFCs
            )
                <= nx.get_edge_attributes(G, "cap")[edge]
        ):
            prob.solution_status = "c2" if debug else -1
            return prob

    # C3: Map 1 VNF - 1 PHYNODE
    for sfc in SFCs:
        for node in G.nodes:
            if not (
                sum(
                    get_solution_value(solution_data, phiNode_S[SFCs.index(sfc)][node_S][node].name)
                    for node_S in sfc.nodes
                )
                <= get_solution_value(solution_data, phiSFC[SFCs.index(sfc)].name)
            ):
                prob.solution_status = "c3" if debug else -1
                return prob

    # C4.1: Map All VNF
    for sfc in SFCs:
        for node_S in sfc.nodes:
            if not (
                sum(
                    get_solution_value(solution_data, phiNode_S[SFCs.index(sfc)][node_S][node].name)
                    for node in G.nodes
                )
                == get_solution_value(solution_data, phiSFC[SFCs.index(sfc)].name)
            ):
                prob.solution_status = "c4" if debug else -1
                return prob

    # C5: Flow-Conservation
    for sfc in SFCs:
        for edge_S in sfc.edges:
            for node in G.nodes:
                if not (
                    sum(
                        get_solution_value(solution_data, str(phiLink_S[SFCs.index(sfc)][edge_S].get((node, nodej))))
                        for nodej in G.nodes
                    )
                    -
                    sum(
                        get_solution_value(solution_data, str(phiLink_S[SFCs.index(sfc)][edge_S].get((nodej, node))))
                        for nodej in G.nodes
                    )
                    == get_solution_value(solution_data, phiNode_S[SFCs.index(sfc)][edge_S[0]][node].name) - get_solution_value(solution_data, phiNode_S[SFCs.index(sfc)][edge_S[1]][node].name)
                ):
                    prob.solution_status = "c5" if debug else -1
                    return prob
    prob.solution_status = "ok" if debug else 1
    return prob


def get_solution_value(solution_data, variable_name):
    return solution_data.get(variable_name, 0)