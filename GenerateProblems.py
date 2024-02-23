import multiprocessing as mp
import os
import uuid
import copy

from utilities.config import ConfigParser
from utilities.multiprocessing import MultiProcessing, IterToQueue
from utilities.dir import RecurseListDir, CleanDir
from GraphMappingProblem import GraphMappingProblem, SaveProblem, LoadProblem
from GraphMappingProblem.utilities import GraphClone

def Main(config:dict):
    print(config)
    PROBLEM_SETNAME = config["PROBLEM_SETNAME"]
    PROBLEM_COUNT = config["PROBLEM_COUNT"]
    OUTPUT_PATH = os.path.join("./data/problems", PROBLEM_SETNAME)
    CleanDir(OUTPUT_PATH)
    KEEPPHY = bool(config["KEEPPHY"])
    PHY_MODE = str(config["PHY"]["MODE"]).split("@")
    SFC_MODE = str(config["SFCSET"]["MODE"]).split("@")
    
    phy_nodecount = config["PHY"]["NODECOUNT"]["min"] if config["PHY"]["NODECOUNT"]["min"] == config["PHY"]["NODECOUNT"]["max"] else (config["PHY"]["NODECOUNT"]["min"], config["PHY"]["NODECOUNT"]["max"])
    phy_nodecap = config["PHY"]["NODECAP"]["min"] if config["PHY"]["NODECAP"]["min"] == config["PHY"]["NODECAP"]["max"] else (config["PHY"]["NODECAP"]["min"], config["PHY"]["NODECAP"]["max"])
    phy_linkcap = config["PHY"]["LINKCAP"]["min"] if config["PHY"]["LINKCAP"]["min"] == config["PHY"]["LINKCAP"]["max"] else (config["PHY"]["LINKCAP"]["min"], config["PHY"]["LINKCAP"]["max"])
    phy_linkdisrate = config["PHY"]["LINKDISRATE"]

    sfc_count = config["SFCSET"]["SFCCOUNT"]
    sfc_nodecount = config["SFCSET"]["NODECOUNT"]["min"] if config["SFCSET"]["NODECOUNT"]["min"] == config["SFCSET"]["NODECOUNT"]["max"] else (config["SFCSET"]["NODECOUNT"]["min"], config["SFCSET"]["NODECOUNT"]["max"])
    sfc_nodereq = config["SFCSET"]["NODEREQ"]["min"] if config["SFCSET"]["NODEREQ"]["min"] == config["SFCSET"]["NODEREQ"]["max"] else (config["SFCSET"]["NODEREQ"]["min"], config["SFCSET"]["NODEREQ"]["max"])
    sfc_linkreq = config["SFCSET"]["LINKREQ"]["min"] if config["SFCSET"]["LINKREQ"]["min"] == config["SFCSET"]["LINKREQ"]["max"] else (config["SFCSET"]["LINKREQ"]["min"], config["SFCSET"]["LINKREQ"]["max"])

    match PHY_MODE[0]:
        case "FROMGML":
            from GraphMappingProblem.phy.fromgml import FromGmlGraphGenerator as PhyGraphGenerator
            phygraphGenerator = PhyGraphGenerator(gml_path=PHY_MODE[1], nodecap=phy_nodecap, linkcap=phy_linkcap)
        case _:
            raise Exception(f"[Invalid config] PHY/MODE={PHY_MODE[0]}")
        
    match SFC_MODE[0]:
        case "linear":
            from GraphMappingProblem.sfc.linear import LinearSfcGraphGenerator as SfcGraphGenerator
            sfcgraphGenerator = SfcGraphGenerator(nodecount=sfc_nodecount, nodereq=sfc_nodereq, linkreq=sfc_linkreq)
        case _:
            raise Exception(f"[Invalid config] SFCSET/MODE={SFC_MODE[0]}")
    
    PHY = phygraphGenerator.Generate()
    for i in range(PROBLEM_COUNT):
        if not KEEPPHY:
            PHY = phygraphGenerator.Generate()
        SFC = sfcgraphGenerator.Generate()
        SFC_SET = []
        for i in range(config["SFCSET"]["SFCCOUNT"]):
            if not config["SFCSET"]["KEEPSFC"]:
                SFC = sfcgraphGenerator.Generate()
            else:
                SFC = GraphClone(SFC)
            SFC_SET.append(SFC)
        problem = GraphMappingProblem(phy=PHY, sfcs=SFC_SET)
        savepath = os.path.join(OUTPUT_PATH, f"{problem.name}.pkl.gz")
        SaveProblem(path=savepath, problem=problem)

def MpWorker(queue: mp.Queue):
    while queue.qsize():
        item = queue.get()
        Main(item)
    exit()

if __name__=="__main__":
    config_list = ConfigParser("./configs/ProblemSettings/dummy.yaml")
    q = IterToQueue(config_list)
    MultiProcessing(MpWorker, (q,), 1)
