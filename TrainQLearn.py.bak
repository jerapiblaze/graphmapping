import GraphMappingProblem
import Solvers.QLearn as QLearn
import gymnasium as gym

import networkx as nx

def Main():
    random_seed = 42 
    PHY = nx.DiGraph()
    nodes = range(0, 7)
    for node in nodes:
        PHY.add_node(node, cap=1000)
    max_edges = 15
    edges = [(i, j) for i in range(0, 7) for j in range(0, 7) if i != j]
    edges = edges[:max_edges] 
    for edge in edges:
        PHY.add_edge(*edge, cap=1000)
        PHY.add_edge(edge[1], edge[0], cap=1000)

    print("\nNode Weights:")
    node_weights = nx.get_node_attributes(PHY, "cap")
    for node, weight in node_weights.items():
        print(f"Node {node}: {weight}")

    # In trọng số của các cạnh
    print("\nEdge Weights:")
    edge_weights = nx.get_edge_attributes(PHY, "cap")
    for edge, weight in edge_weights.items():
        print(f"Edge {edge}: {weight}")      

    # SFC_SET = graph_generator.flex_sfc_set2.PfoSFCSET(sfc_count=2, node_count_params=[5, 6, 5], node_req_params=[10, 50, 1], link_req_params=[10, 50, 1], flex_rate_params=0, seed = random_seed)
    SFC_SET = [GraphMappingProblem.sfc.linear.LinearSfcGraphGenerator(nodecount=4, nodereq=10,linkreq=10).Generate() for i in range(5)]
    print(SFC_SET[0].nodes(data = True))
    print(SFC_SET[0].edges(data = True))
    # for  sfc in enumerate(SFC_SET):

    #     # In thông tin về các nút
    #     print("\nNodes:")
    #     for node, data in sfc.nodes(data=True):
    #         print(f"Node {node}: {data}")
    #     # In thông tin về các cạnh và trọng số
    #     print("\nEdges:")
    #     for edge in sfc.edges(data=True):
    #         print(f"Edge {edge}")

    #     print("\n")    
    # problem = GraphMappingProblem.LoadProblem(".\data\problems\DUMMY\graphmapping_a0662021.pkl.gz")
    # print(problem.PHY)
    env = QLearn.env.StaticMapping2Env(PHY, SFC_SET, {"node_req": "req", "link_req": "req", "node_cap": "cap", "link_cap": "cap"}, 2000, 10)
    
    # def get_node_cap(node_id):
    #     node_caps = nx.get_node_attributes(PHY, name="node_cap")
    #     print("node_caps:", node_caps)
    #     if (node_id is None):
    #         return node_caps
    #     return node_caps[node_id]
    # a = get_node_cap(6)
    # print(a)
    agent = QLearn.agent.QLearningAgent(env.obs_space_size, env.action_space_size, alpha=0.1, gamma=0.9, epsilon=0.1)
    trained_agent, q_values = QLearn.agent.TrainAgent(agent, env, 100, True)
    with open("./debug_q.csv", "wt") as f:
        f.write("ep,q\n")
        for q in q_values:
            f.write(f"{q[0]},{q[1]}\n")
    # QLearn.agent.SaveAgent("./data/__internals__/QLearn/DUMMY.pkl.gz", trained_agent)
    pass

if __name__=="__main__":
    Main()
    pass