import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from Solvers.BnBonk.model.graph import Phy, Sfc
from Solvers.BnBonk.model.solution import SfcSolution
from Solvers.BnBonk.model.embed import EmbSfc
from Solvers.BnBonk.helper.concatnodes import ConcatNodes
from Solvers.BnBonk.helper.helper import GetPair, getTotalReq


def MapSlice(
        sfc: Sfc,
        phy: Phy,
        count_node: int,
        cur_node_id: int,
        solution: SfcSolution,
        emb: EmbSfc,
):
    
    if count_node == sfc.vnf_nums():
        if solution.updateSolution(getTotalReq(solution.cur_sol())):
            solution.official_solution()
        return
    
    for node in phy.nodes():
        # Check if a node in the physical graph is mapped by another node?
        if emb.is_exists_node(node):
            continue
        # Check resource conditions?
        if sfc.node_weight(count_node) > phy.node_weight(node[0]):
            continue
        # Find path between previous node and current node
        mapping_links = []
        edges = []
        if not count_node == 0:
            prev_node_mapping = GetPair(solution.cur_sol(), count_node - 1)
            if prev_node_mapping is None:
                continue
            edge_prev_cur = sfc.edge_between_node(count_node - 1, count_node)
            if edge_prev_cur is None:
                continue
            path = ConcatNodes(phy, emb.links(), prev_node_mapping, node[0], edge_prev_cur)
            if path is None:
                continue
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            for edge in edges:
                link = {
                    f"xEdge_{sfc.id()}_({sfc.nodes_order()[cur_node_id - 1]},{sfc.nodes_order()[cur_node_id]})_({edge[0]},{edge[1]})": 1}
                solution.add_current(link)
                mapping_links.append(link)
                emb.add_link(edge)
        # backtracking part ?
        mapping_node = {f"xNode_{sfc.id()}_{sfc.nodes_order()[cur_node_id]}_{node[0]}": 1}
        solution.add_current(mapping_node)
        emb.add_node(node)
        MapSlice(
            sfc=sfc,
            phy=phy,
            count_node=count_node + 1,
            cur_node_id= cur_node_id+ 1,
            solution=solution,
            emb=emb,
        )
        emb.remove_node(node)
        emb.remove_links(edges)
        solution.remove_current()
        solution.remove_current_links(mapping_links)
