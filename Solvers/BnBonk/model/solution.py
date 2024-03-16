import copy as cp

class SfcSolution:
    def __init__(self, id: int) -> None:
        self.__current_solution = []
        self.__solutions = []
        self.__id = id
        self.__last_cost = -1
    
    def add_current(self, pair: dict[int:int]):
        self.__current_solution.append(pair)
        
    def remove_current(self):
        self.__current_solution = self.__current_solution[:-1]
    
    def official_solution(self):
        tmp_result = {}
        tmp_cur_sol = cp.deepcopy(self.__current_solution)
        for pair in tmp_cur_sol:
            tmp_result.update(pair)
            ld = list(pair.keys())
            ld_str = ld[0].split('_')
            if ld_str[0] == "xEdge":
                key = f"y_{ld_str[1]}_{ld_str[2]}"
                if not key in tmp_result:
                    tmp_result.update({key: 1})
        
        self.__solutions.append(tmp_result)
    
    def updateSolution(self, cost) -> bool:
        if self.__last_cost == -1 or self.__last_cost > cost:
            self.__last_cost = cost
            return True
        return False
        
    def cur_sol(self) -> list[dict[int:int]]:
        return self.__current_solution
    
    def is_empty(self) -> bool:
        if len(self.__solutions) == 0:
            return True
        return False
    
    def sol(self) -> list:
        return self.__solutions
    
    def segment_id(self) -> int:
        return self.__id

    def remove_current_links(self, mapping_links: list[dict]):
        self.__current_solution = [item for item in self.__current_solution if item not in mapping_links]
    
    def optimal(self) -> dict:
        return self.__solutions[-1]


class Solution:
    def __init__(self) -> None:
        self.__solution = {}
        self.__cur_sol = []
    
    def update(self, solution_sfc, sfc_id: int):
        tmp_solution_sfc = cp.deepcopy(solution_sfc)
        element_solution = {}
        for sol in tmp_solution_sfc:
            element_solution.update({sol:1})
        element_solution.update({f"xSFC_{sfc_id}": 1})
        self.__cur_sol.append(element_solution)
        return
    
    def official(self):
        tmp_cur_sol = cp.deepcopy(self.__cur_sol[-1])
        for sol in tmp_cur_sol:
            self.__solution.update({sol:1})
    
    def get_last_sol(self) -> dict:
        return self.__cur_sol[-1]
    
    def sol_to_validate(self) -> dict:
        return self.__solution