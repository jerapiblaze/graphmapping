import networkx as nx
import copy as cp

class Sfc:
    def __init__(self, sfc: nx.DiGraph, sfc_id: int) -> None:
        self.__nodes_order = list(sfc.nodes())
        self.__nodes = list(sfc.nodes(data=True))
        self.__edges = list(sfc.edges(data=True))
        self.__edges_weight = nx.get_edge_attributes(sfc, 'req')
        self.__nodes_weight = nx.get_node_attributes(sfc, 'req')
        self.__id = sfc_id
        self.__req_link = 0
        
    def set_req_link(self, req_segment_link: int):
        self.__req_link = req_segment_link
    
    def get_req_link(self) -> int:
        return self.__req_link
    
    def nodes(self) -> list:
        return self.__nodes
    
    def edges(self) -> list:
        return self.__edges
    
    def id(self) -> int:
        return self.__id
    
    def node_weight(self, id_node: int):
        return self.__nodes[id_node][1]['req']
    
    def edge_between_node(self, id_node_source, id_node_target):
        node_1 = self.__nodes[id_node_source][0]
        node_2 = self.__nodes[id_node_target][0]
        for edge in self.__edges:
            if node_1 == edge[0] and node_2 == edge[1]:
                return edge[2]['req']
        return None
    
    def node_name(self, id_node: int) -> int:
        return self.__nodes()[id_node][0]
    
    def vnf_nums(self) -> int:
        return len(self.__nodes)
    
    def first_node_id(self) -> int:
        return self.__nodes[0][0]
    
    def last_node(self):
        return self.__nodes[-1][0]
    
    def get_nodes_weight(self) -> dict[int:int]:
        return self.__nodes_weight
    
    def get_edges_weight(self) -> dict[set:int]:
        return self.__edges_weight
    
    def nodes_order(self) -> list:
        return self.__nodes_order


class Phy:
    def __init__(self, PHY) -> None:
        self.__nodes = list(PHY.nodes(data=True))
        self.__edges = list(PHY.edges(data=True))
        self.__nodes_weight = nx.get_node_attributes(PHY, 'cap')
        self.__edges_weight = nx.get_edge_attributes(PHY, 'cap')

    def nodes(self) -> list:
        return self.__nodes

    def edges(self) -> list:
        return self.__edges

    def update_new_nodes(self, v_nodes: dict[int:int], p_nodes: dict[int:int]):
        for key, value in p_nodes.items():
            self.__nodes_weight[key] = self.__nodes_weight[key] - v_nodes[value]

    def update_new_edges(self, v_links: dict[set:int], p_links: dict[set:int]):
        for key, value in p_links.items():
            self.__edges_weight[key] = self.__edges_weight[key] - v_links[value]

    def update_new(self, result: dict[str:int], sfc: Sfc):
        p_nodes_dict = {}
        p_links_dict = {}
        v_links_set = set()
        keys = list(result.keys())
        for key in keys:
            k = key.split('_')
            if k[0] == 'xNode':
                p_nodes_dict.update({int(k[3]): int(k[2])})
            if k[0] == 'xEdge':
                p_link = tuple(k[3].removeprefix("(").removesuffix(")").split(","))
                v_link = tuple(k[2].removeprefix("(").removesuffix(")").split(","))
                p_links_dict.update({(int(p_link[0]), int(p_link[1])): (int(v_link[0]), int(v_link[1]))})
                v_links_set.add((int(v_link[0]), int(v_link[1])))
        v_nodes_weight = sfc.get_nodes_weight()
        v_links_weight = sfc.get_edges_weight()
        self.update_new_nodes(v_nodes_weight, p_nodes_dict)
        self.update_new_edges(v_links_weight, p_links_dict)

    def update_old(self):
        return

    def sub_phy(self, is_map_link_phy: list, req_segment_link: int) -> nx.DiGraph:
        node_set = self.__nodes
        edge_set = cp.deepcopy(self.__edges)

        sub_phy = nx.DiGraph()
        if not len(is_map_link_phy) == 0:
            for edge in is_map_link_phy:
                for value in edge_set[:]:
                    if str(edge[0]) == str(value[0]) and str(edge[1]) == str(value[1]):
                        edge_set.remove(value)
                        break

        # Remove link have weight < req_segment_link
        for edge in edge_set[:]:
            if edge[2]['cap'] < req_segment_link:
                edge_set.remove(edge)

        for edge in edge_set:
            sub_phy.add_edge(int(edge[0]), int(edge[1]), cap=edge[2]['cap'])

        node_sub_graph_weights = {int(node[0]): node[1]['cap'] for node in node_set}
        nx.set_node_attributes(sub_phy, node_sub_graph_weights, 'cap')
        return sub_phy

    def node_weight(self, node_id: int) -> int:
        return self.__nodes_weight[node_id]

    def nodes_set_weight(self) -> dict:
        return self.__nodes_weight

    def edges_set_weight(self) -> dict:
        return self.__edges_weight
