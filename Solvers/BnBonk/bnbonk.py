import sys
import os

from .model.graph import Phy, Sfc
from .BnB.sfcSet import MapSFCs

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def BnBStar(PHY, SFC_SET, profiler: StopWatch):
    profiler.add_stop(f"se_bnbstar_v5: PHY({len(PHY.nodes)}  nodes, {len(PHY.edges)} links), {len(SFC_SET)} sfcs")
    phy = Phy(PHY)
    i = 0
    sfcSet = []
    for sfcBefore in SFC_SET:
        sfc = Sfc(sfcBefore, i)
        i = i + 1
        sfcSet.append(sfc)
    sol = MapSFCs(sfcSet, phy)
    return sol.sol_to_validate()
