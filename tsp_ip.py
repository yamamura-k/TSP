import pulp
from math import isclose
from typing import List
def solve(ncity: int, D: List[float]) -> List[int]:
    cities = list(range(ncity))
    x = pulp.LpVariable.dicts("x", (cities, cities), cat="Binary")# 都市iから都市jに向かうかを表す0-1変数
    u = pulp.LpVariable.dicts("u", cities, cat="Integer", lowBound=1, upBound=ncity-1)
    problem = pulp.LpProblem("TSP_IP")
    problem += pulp.lpSum(D[i][j]*x[i][j] for i in cities for j in cities)
    for i in cities:
        problem += pulp.lpSum(x[i][j] for j in cities) == 1# 移動先は一つの都市
        problem += x[i][i] == 0# 同じ都市に止まることはNG
        for j in cities:# 部分巡回除去制約(MTZ条件において、w_i=0とした)
            if i == j:
                continue
            problem += u[i] - u[j] + ncity*x[i][j] <= ncity - 1
    for j in cities:
        problem += pulp.lpSum(x[i][j] for i in cities) == 1# 移動元は一つの都市
    status = problem.solve(pulp.CPLEX_CMD(msg=1))
    tour = []
    for i in cities:
        for j in cities:
            if isclose(x[i][j].value(), 1):
                tour.append((i, j))
                break
    #assert isclose(x[tour[-1]][tour[0]].value(), 1)
    #assert len(tour) == ncity
    print(tour)

    return tour
