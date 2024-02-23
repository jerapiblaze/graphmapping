import networkx as nx
import uuid
import copy

def GraphClone(target: nx.DiGraph):
    cloned = copy.deepcopy(target)
    cloned.name = f"{target.name}_cloned_{uuid.uuid4().hex[:8]}"
    return cloned