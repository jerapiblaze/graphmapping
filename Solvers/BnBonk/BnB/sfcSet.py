import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from Solvers.BnBonk.BnB.bonkSfc import MapSlice
from Solvers.BnBonk.model.solution import Solution, SfcSolution
from Solvers.BnBonk.model.graph import Phy, Sfc
from Solvers.BnBonk.model.embed import EmbSfc
# from utilities.profiler import StopWatch


def MapSFCs(
        sfc_set: list[Sfc],
        phy: Phy,
) -> Solution:
    
    solution = Solution()
    i = 0
    for sfc in sfc_set:
        count = 0
        curNodeId = 0
        solutionSfc = SfcSolution(sfc.id())
        embSfc = EmbSfc()
        MapSlice(
            sfc=sfc,
            phy=phy,
            count_node=count,
            cur_node_id=curNodeId,
            solution=solutionSfc,
            emb=embSfc,
        )
        if solutionSfc.is_empty():
            continue
        solution.update(solutionSfc.optimal(), sfc.id())
        phy.update_new(solution.get_last_sol(), sfc)
        print(f"Done Slice {i}")
    solution.get_last_sol()
    
    return solution
