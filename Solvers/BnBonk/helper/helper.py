import networkx as nx


def GetPair(pairs_list: list[dict[str:int]], target) -> int:
    for pair in pairs_list:
        pair_key = list(pair.keys())[0].split('_')
        if not pair_key[0] == 'xNode':
            continue
        if int(pair_key[2]) == target:
            return int(pair_key[3])
    return None


def SegmentSet(sfc: nx.DiGraph):
    subseg_nodes_set = nx.weakly_connected_components(sfc)
    subseg = []
    for subseg_nodes in subseg_nodes_set:
        subseg.append(nx.subgraph(sfc, subseg_nodes))
    return subseg

def getTotalReq(sol: dict, sfcSet) -> int:
    ansNode = 0
    ansEdge = 0
    keys = list(sol.keys())
    keysSplit = [k.split('_') for k in keys]
    nodesSet = [nx.get_node_attributes(sfc, 'req') for sfc in sfcSet]
    edgesSet = [nx.get_edge_attributes(sfc, 'req') for sfc in sfcSet]

    for key in keysSplit:
        sfcId = int(key[1])
        if key[0] == "xNode":
            sfcNode = int(key[2])
            ansNode = ansNode + nodesSet[sfcId][sfcNode]
        if key[0] == "xEdge":
            sfcEdge = key[2]
            sfcEdge = tuple(sfcEdge.removeprefix("(").removesuffix(")").split(","))
            sfcEdge = (int(sfcEdge[0]), int(sfcEdge[1]))
            ansEdge = ansEdge + edgesSet[sfcId][sfcEdge]

    return ansNode + ansEdge
