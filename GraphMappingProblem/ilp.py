from .__internals__ import *
from pulp import *

VARIABLE_SURFIXES = ["xNode","xEdge","xSFC"]

def ConvertToIlp(problem:GraphMappingProblem) -> pulp.LpProblem:
    G = problem.PHY
    SFCs = problem.SFC_SET
    name = problem.name
    __problem = LpProblem(name=name, sense=LpMinimize)

    # Build Node Placement List
    phiNode_S = list()
    for sfc in SFCs:
        phiNode_S.append(
            LpVariable.dicts(
                name=f"xNode_{SFCs.index(sfc)}",
                indices=(sfc.nodes, G.nodes),
                cat = "Binary"
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

    # Bulding Constraints
    __problem.constraints.clear()

    ## C1: Node Capacity
    for node in G.nodes:
        __problem += (
            lpSum(
                lpSum(
                    phiNode_S[SFCs.index(sfc)][node_S][node] * nx.get_node_attributes(sfc, "req")[node_S]
                        for node_S in sfc.nodes
                ) 
                for sfc in SFCs
            )
                <= nx.get_node_attributes(G, "cap")[node], 
                f"C1_i{node}"
        )

    ## C2: Edge Capacity
    for edge in G.edges:
        __problem += (
            lpSum(
                lpSum(
                    phiLink_S[SFCs.index(sfc)][link_S][edge] * nx.get_edge_attributes(sfc, "req")[link_S]
                    for link_S in sfc.edges
                ) 
                for sfc in SFCs
            )
                <= nx.get_edge_attributes(G, "cap")[edge], 
            f"C2_ij{edge}"
        )

    ## C3: Map 1 VNF - 1 PHYNODE
    for sfc in SFCs:
        for node in G.nodes:
            __problem += (
                lpSum(
                    phiNode_S[SFCs.index(sfc)][node_S][node]
                    for node_S in sfc.nodes
                )
                <= 1
                ,
                f"C3_i{node}_s{SFCs.index(sfc)}"
            )

    ## C4.1: Map All VNF
    for sfc in SFCs:
        for node_S in sfc.nodes:
            __problem += (
                lpSum(
                    phiNode_S[SFCs.index(sfc)][node_S][node]
                    for node in G.nodes
                )
                == phiSFC[SFCs.index(sfc)]
                ,
                f"C4_v{node_S}_s{SFCs.index(sfc)}"
            )

    ## C4.2: Map All VLinks
    for sfc in SFCs:
        for edge_S in sfc.edges:
            __problem += (
                lpSum(
                    phiLink_S[SFCs.index(sfc)][edge_S][edge]
                    for edge in G.edges
                )
                == phiSFC[SFCs.index(sfc)],
                f"C4_v{edge_S}_s{SFCs.index(sfc)}"
            )

    ## C5: Flow-Conservation
    for sfc in SFCs:
        for edge_S in sfc.edges:
            for node in G.nodes:
                __problem += (
                    lpSum(
                        phiLink_S[SFCs.index(sfc)][edge_S].get((node, nodej))
                        for nodej in G.nodes
                    ) 
                    - 
                    lpSum(
                        phiLink_S[SFCs.index(sfc)][edge_S].get((nodej, node))
                        for nodej in G.nodes
                    )
                    == phiNode_S[SFCs.index(sfc)][edge_S[0]][node] - phiNode_S[SFCs.index(sfc)][edge_S[1]][node]
                    ,
                    f"C5_s{SFCs.index(sfc)}_vw{edge_S}_i{node}"
                )

    # Target function building
    __problem += (
        0 - lpSum(
            phiSFC[SFCs.index(sfc)]
            for sfc in SFCs
        )
    )

    return __problem