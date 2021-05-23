import pulp
from math import isclose
from typing import List
class PulpIP:
    def __init__(self, ncity: int, D: List[float], MTZ_level: int =0) -> None:
        self.ncity = ncity
        self.D = D
        self.cities = list(range(ncity))
        self.x = pulp.LpVariable.dicts("x", (self.cities, self.cities), cat="Binary")# 都市iから都市jに向かうかを表す0-1変数
        self.u = pulp.LpVariable.dicts("u", self.cities, cat="Integer", lowBound=0, upBound=ncity-1)# 訪問順序を表す
        self.problem = pulp.LpProblem("TSP_IP")
        self.problem += pulp.lpSum(self.D[i][j]*self.x[i][j] for i in self.cities for j in self.cities)
        for i in self.cities:
            self.problem += pulp.lpSum(self.x[i][j] for j in self.cities) == 1# 移動先は一つの都市
            self.problem += pulp.lpSum(self.x[j][i] for j in self.cities) == 1# 移動元は一つの都市
            self.problem += self.x[i][i] == 0# 同じ都市に止まることはNG
        self.set_MTZ(level=MTZ_level)

        self.problem += self.u[self.cities[0]] == 0
        for i in self.cities[1:]:
            self.problem += self.u[i] >= 1
    
    def set_MTZ(self, level: int) -> None:
        if level == 0:
            for i in self.cities:
                for j in self.cities[1:]:# 部分巡回除去制約(MTZ条件)
                    if i == j:
                        continue
                    self.problem += self.u[i] - self.u[j] + self.ncity*self.x[i][j] <= self.ncity - 1
        elif level >= 1:
            for i in self.cities[1:]:
                for j in self.cities[1:]:# 部分巡回除去制約ちょっと強め(MTZ条件)
                    if i == j:
                        continue
                    # self.problem += self.u[i] + 1 - (self.ncity - 1)*(1 - self.x[i][j]) + (self.ncity - 3)*self.x[j][i] <= self.u[j]
                    self.problem += self.u[i] - self.u[j] + (self.ncity - 1)*self.x[i][j] + (self.ncity - 3)*self.x[j][i] <= self.ncity - 2
            if level >= 2:# 部分巡回除去制約、強め(MTZ条件)
                for i in self.cities[1:]:
                    self.problem += 2 - self.x[0][i] + (self.ncity - 3)*self.x[i][0] <= self.u[i]
                    self.problem += self.u[i] <= (self.ncity - 1) - (1 - self.x[i][0]) - (self.ncity - 3)*self.x[0][i]


    def solve(self, solver_name: str ="cbc", initial_tour: List[int] =None, threads: int=2) -> List[int]:
        ws = False
        if initial_tour:
            self.warmStart(initial_tour)
            ws = True
        if solver_name == "cbc":
            solver = pulp.PULP_CBC_CMD(msg=0, warmStart=ws, threads=threads)
        elif solver_name == "cplex":
            solver = pulp.CPLEX_CMD(msg=0, threads=threads)
        else:
            print(f"Cannot use {solver_name} to solve.")
            print("We use cbc solver instead.")
        try:
            status = self.problem.solve(solver)
        except:
            solver = pulp.PULP_CBC_CMD(msg=0, warmStart=ws, threads=threads)
            status = self.problem.solve(solver)

        tour = list(self.cities)
        tour.sort(key=lambda x:self.u[x].value())
        for i, j in zip(tour, tour[1:]+tour[:1]):
            assert isclose(self.x[i][j].value(), 1)
        assert len(tour) == self.ncity
    
        return tour
    
    def warmStart(self, tour: List[int]) -> None:
        index = dict()
        for i in range(self.ncity):
            index[tour[i]] = i
        for i in self.cities:
            for j in self.cities:
                if index[i] - index[j] == 1:
                    self.x[i][j].setInitialValue(1)
                else:
                    self.x[i][j].setInitialValue(0)