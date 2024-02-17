from .__internals__ import *

class FromGmlGraphGenerator(PhysicalGraphGenerator):
    def __init__(self, gml_path:str, nodecap:int|tuple[int,int], linkcap:int|tuple[int,int]) -> None:
        self.__graph__ = nx.DiGraph(nx.read_gml(gml_path))
        self.nodecap = nodecap
        self.linkcap = linkcap
        self.basename = os.path.basename(gml_path)

    def Generate(self) -> nx.DiGraph:
        nodes = list(self.__graph__.nodes)
        links = [(nodes.index(e[0]), nodes.index(e[1])) for e in list(self.__graph__.edges)]
        PHY_nodes = [(nodes.index(node),{"cap":self.nodecap if type(self.nodecap) == int else rd.randint(self.nodecap[0], self.nodecap[1])}) for node in nodes]
        PHY_links = [(link[0], link[1],{"cap":self.linkcap if type(self.linkcap) == int else rd.randint(self.linkcap[0], self.linkcap[1])}) for link in links]
        PHY = nx.DiGraph(name=f"{self.basename}_{len(PHY_nodes)}nodes_{len(PHY_links)}_{uuid.uuid4().hex[:8]}")
        PHY.add_nodes_from(PHY_nodes)
        PHY.add_edges_from(PHY_links)
        return PHY