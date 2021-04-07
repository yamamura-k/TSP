import pulp
from math import isclose
from typing import List
class PulpIP:
    def __init__(self, ncity: int, D: List[float]) -> None:
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
            for j in self.cities[1:]:# 部分巡回除去制約(MTZ条件)
                if i == j:
                    continue
                self.problem += self.u[i] - self.u[j] + ncity*self.x[i][j] <= ncity - 1

        self.problem += self.u[self.cities[0]] == 0
        for i in self.cities[1:]:
            self.problem += self.u[i] >= 1

    def solve(self, solver_name: str ="cbc", initial_tour: List[int] =None) -> List[int]:
        ws = False
        if initial_tour:
            self.warmStart(initial_tour)
            ws = True
        if solver_name == "cbc":
            solver = pulp.PULP_CBC_CMD(msg=0, warmStart=ws)
        elif solver_name == "cplex":
            solver = pulp.CPLEX_CMD(msg=0)
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