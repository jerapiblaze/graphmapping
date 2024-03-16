import sys
import os
import typing as T
import networkx as nx
import copy as cp

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from BnBonk.model.graph import Phy


def get_edge_weight(u,v, weight_dic):
    weight = 100000 - weight_dic["cap"]
    return weight


def find_path(sub_phy, source, target):
    try:
        path = nx.shortest_path(sub_phy, int(source), int(target), get_edge_weight)
        return path
    except nx.NetworkXNoPath:
        return None
    except nx.NodeNotFound:
        return None


def ConcatNodes(phy: Phy,
             is_map_link_phy: list,
             source: int, target: int,
             link_require: int):
    node_set = cp.deepcopy(phy.nodes()) 
    edge_set = cp.deepcopy(phy.edges())
        
    if not len(is_map_link_phy) == 0:
        for edge in is_map_link_phy:
            for value in edge_set[:]:
                if edge[0] == value[0] and edge[1] == value[1]:
                    edge_set.remove(value)
        
        for edge in edge_set[:]:
            if edge[2]['cap'] < link_require:
                edge_set.remove(edge)
    
    sub_phy = nx.DiGraph()
    for edge in edge_set:
        sub_phy.add_edge(edge[0], edge[1], cap = edge[2]['cap'])

    node_sub_graph_weights = {}
    for node in node_set:
        node_sub_graph_weights.update({node[0]: node[1]['cap']})
    nx.set_node_attributes(sub_phy, node_sub_graph_weights, 'cap')

    path, path_len = find_path(sub_phy, source, target)
    return path

