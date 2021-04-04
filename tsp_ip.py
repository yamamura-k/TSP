import pulp
from math import isclose
from typing import List
def solve(ncity: int, D: List[float]) -> List[int]:
    cities = list(range(ncity))
    x = pulp.LpVariable.dicts("x", (cities, cities), cat="Binary")# 都市iから都市jに向かうかを表す0-1変数
    problem = pulp.LpProblem("TSP_IP")
    problem += pulp.lpSum(D[i][j]*x[i][j] for i in cities for j in cities)
    for i in cities:
        problem += pulp.lpSum(x[i][j] for j in cities) == 1# 移動先は一つの都市
        problem += pulp.lpSum(x[j][i] for j in cities) == 1# 移動元は一つの都市
        problem += x[i][i] == 0# 同じ都市に止まることはNG
        for j in cities:# 部分巡回除去制約(MTZ条件において、w_i=0とした)
            problem += ncity*x[i][j] <= ncity - 1
    status = problem.solve()
    tour = [0]
    for _ in range(ncity):
        now = tour[-1]
        for j in cities:
            if isclose(x[now][j].value(), 1):
                tour.append(j)
                break
    assert isclose(x[tour[-1]][0].value(), 1)
    assert len(tour) == ncity
    return tour