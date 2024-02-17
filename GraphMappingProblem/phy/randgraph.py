from .__internals__ import *

class RandomGraphGenerator(PhysicalGraphGenerator):
    def __init__(self, nodecap: int | tuple[int, int], linkcap: int | tuple[int, int], nodecount:int|tuple[int, int], linkdisconnectrate:float=0) -> None:
        self.nodecap = nodecap
        self.linkcap = linkcap
        self.nodecount = nodecount
        self.linkdisconnectrate = linkdisconnectrate

    def Generate(self) -> nx.DiGraph:
        nodecount = self.nodecount if type(self.nodecount) == int else rd.randint(self.nodecount[0], self.nodecount[1])
        PHY = nx.DiGraph()
        for i in range(nodecount):
            nodecap = self.nodecap if type(self.nodecap) == int else rd.randint(self.nodecap[0], self.nodecap[1])
            PHY.add_node(i, cap=nodecap)
        for n in PHY.nodes:
            for nn in PHY.nodes:
                if n == nn:
                    continue
                if (PHY.edges.get((n, nn), None)):
                    continue
                linkcap = self.linkcap if type(self.linkcap) == int else rd.randint(self.linkcap[0], linkcap[1])
                PHY.add_edge(n, nn, cap=linkcap)
                linkcap = self.linkcap if type(self.linkcap) == int else rd.randint(self.linkcap[0], linkcap[1])
                PHY.add_edge(nn, n, cap=linkcap)
        linkcount = len(list(PHY.edges))
        for i in range(int(linkcount * self.linkdisconnectrate)):
            link = rd.choice(list(PHY.edges))
            PHY.remove_edge(link[0], link[1])
        PHY.name = f"randomphy_{len(list(PHY.nodes))}nodes_{len(list(PHY.edges))}links_{uuid.uuid4().hex[:8]}"
        return PHY
