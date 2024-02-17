from .__internals__ import *

class LinearSfcGraphGenerator(SfcGraphGenerator):
    def __init__(self, nodecount:int|tuple[int,int], nodereq:int|tuple[int,int], linkreq:int|tuple[int,int]):
        self.nodecount = nodecount
        self.nodereq = nodereq
        self.linkreq = linkreq

    def Generate(self) -> nx.DiGraph:
        SFC = nx.DiGraph()
        nodecount = self.nodecount if type(self.nodecount) == int else rd.randint(self.nodecount[0], self.nodecount[1])
        for i in range(nodecount):
            nodereq = self.nodereq if type(self.nodereq) == int else rd.randint(self.nodereq[0], self.nodereq[1])
            SFC.add_node(i, req=nodereq)
        for n in list(SFC.nodes)[:-1]:
            linkreq = self.linkreq if type(self.linkreq) == int else rd.randint(self.linkreq[0], self.linkreq[1])
            SFC.add_edge(n, n+1, req=linkreq)
        SFC.name = f"linearsfc_{len(list(SFC.nodes))}nodes_{len(list(SFC.edges))}links_{uuid.uuid4().hex[:8]}"
        return SFC