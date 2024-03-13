import networkx as nx
import numpy as np

def Get_PHY_matrix(graph1:nx.DiGraph, nodecapname:str, linkcapname:str) -> np.ndarray:
    A_i = np.array(nx.adjacency_matrix(graph1, weight=linkcapname).todense())

    for i in range(len(A_i)):
        for j in range(len(A_i[i])):
            if A_i[i][j] > 0:
                continue
            if i == j:
                cap = nx.get_node_attributes(graph1, name=nodecapname).get(i, 0)
                A_i[i][j] = cap
                continue
            bw, path = GetMaxIndirectBandwidth(graph1, i, j, linkcapname)
            A_i[i][j] = bw
    return A_i

def GetMaxIndirectBandwidth(graph, start, destination, capname):
    # return GetMaxIndirectBandwidth_bruteforce(graph, start, destination)
    return GetMaxIndirectBandwidth_widestpath(graph, start, destination, capname)

def GetPathListFromPath(path):
    return [(a, b) for a in path for b in path if path.index(b)-path.index(a) == 1]

def GetPathFromHops(pathlist):
    l = []
    l.append(pathlist[0][0])
    l.append(pathlist[0][1])
    for path in pathlist[1:]:
        l.append(path[1])
    return l

def GetMaxIndirectBandwidth_widestpath(Graph, src, target, capname:str):
    ver_list = []
    # To keep track of widest distance
    widest = [-10**9]*(len(Graph)+1)

    # To get the path at the end of the algorithm
    parent = [-10**9]*(len(Graph)+1)

    container = []
    container.append((0, src))

    widest[src] = 10**9
    container = sorted(container)

    while (len(container) > 0):
        temp = container[-1]
        current_src = temp[1]
        del container[-1]

        for neighbor in Graph.neighbors(current_src):

            weight = Graph[current_src][neighbor][capname]
            weight_u_v = weight
            v = neighbor
            u = current_src

            distance = max(widest[v], min(widest[u], weight_u_v))

            # Relaxation of edge and adding into Priority Queue
            if (distance > widest[v]):
                # Updating bottle-neck distance
                widest[v] = distance

                # To keep track of parent
                parent[v] = current_src

                # Adding the relaxed edge in the priority queue
                container.append((distance, v))
                container = sorted(container)

    path = []
    printpath(parent, target, target, path)

    bw = widest[target]
    if bw < 0:
        path = None
        bw = 0
    return bw, path

def printpath(parent, vertex, target, path):
    # global parent
    if (vertex < 0):
        return
    printpath(parent, parent[vertex], target, path)
    path.append(vertex)