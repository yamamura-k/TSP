from typing import List
class TSP_DP:
    def __init__(self, ncity: int, D: List[float]) -> None:
        self.ncity = ncity
        self.D = D
    def solve(self):
        if self.ncity > 16:
            raise ValueError("This problem is too large to solve in practical time.")
        S_max = 1 << self.ncity
        prev_node = [[None]*self.ncity for _ in range(S_max)]
        dp = [[float('inf')]*self.ncity for _ in range(S_max)]
        dp[0][0] = 0
        for S in range(S_max):
            for v in range(self.ncity):
                if S&(1<<v):
                    for j in range(self.ncity):
                        candidate = dp[S-(1<<v)][j] + self.D[j][v]
                        if candidate < dp[S][v]:
                            dp[S][v] = candidate
                            prev_node[S][v] = (j, S-(1<<v))
        tour = []
        index, subset = 0, S_max - 1
        while prev_node[subset][index] is not None:
            tour.append(index)
            index, subset = prev_node[subset][index]
        
        return tour

if __name__=="__main__":
    from utils import read, calc_dist
    filename = "./ALL_tsp/ulysses16.tsp"
    name, ncity, D, coord = read(filename)
    problem = TSP_DP(ncity, D)
    tour = problem.solve()
    tour_len = calc_dist(tour, D)
    print(tour_len)
    print(*tour)