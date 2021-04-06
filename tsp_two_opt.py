import random
from typing import List, Tuple
class TwoOpt:
    def __init__(self, ncity: int, D: List[float]) -> None:
        self.ncity = ncity
        self.D = D
        self.current_tour = None
        self.best_tour = None
        self.current_obj = None
        self.best_obj = None

    def initial_tour(self, strategy: str =None) -> Tuple[List[int], float]:
        if strategy == "greedy":
            return self._greedy()
        elif strategy == "greedy_random":
            return self._greedy(random.randint(1, self.ncity + 1))
        elif strategy == "random":
            tour = random.shuffle(list(range(1, self.ncity + 1)))
            return tour, self.calc_score(tour)
        elif strategy == "convexhull":
            raise NotImplementedError
        else:
            tour = list(range(1, self.ncity + 1))
            return tour, self.calc_score(tour)
    
    def _greedy(self, s=1):
        tour = [s]
        obj = 0
        for _ in range(self.ncity - 1):
            dist = float('inf')
            nxt = None
            for j in range(1, self.ncity + 1):
                if j in tour:
                    continue
                if self.D[tour[-1] - 1][j - 1] < dist:
                    dist = self.D[tour[-1] - 1][j - 1]
                    nxt = j
            tour.append(nxt)
            obj += dist
        obj += self.D[tour[-1] - 1][tour[0] - 1]
        return tour, obj

    def swap_cost(self, i: int, j: int) -> float:
        i_now = self.current_tour[i] - 1
        i_prev = self.current_tour[i - 1] - 1
        i_next = self.current_tour[(i + 1)%self.ncity] - 1 

        j_now = self.current_tour[j] - 1
        j_prev = self.current_tour[j - 1] - 1
        j_next = self.current_tour[(j + 1)%self.ncity] - 1
        
        current_cost = self.D[i_prev][i_now] + self.D[i_now][i_next] + self.D[j_prev][j_now] + self.D[j_now][j_next]
        if j_now == i_next:
            i_next = i_now
            j_prev = j_now
        if i_now == j_next:
            j_next = j_now
            i_prev = i_now
        new_cost = self.D[i_prev][j_now] + self.D[j_now][i_next] + self.D[j_prev][i_now] + self.D[i_now][j_next]
        return new_cost - current_cost
    
    def swap(self, i: int, j: int) -> None:
        _swap_cost = self.swap_cost(i, j)
        if _swap_cost < 0:
            self.current_tour[i], self.current_tour[j] = self.current_tour[j], self.current_tour[i]
            self.current_obj += _swap_cost
            self.best_tour = list(self.current_tour)
            self.best_obj = self.current_obj
    
    def calc_score(self, tour):
        return sum(self.D[i - 1][j - 1] for i, j in zip(tour, tour[1:] + tour[:1]))
    
    def solve_two_opt(self, MAXSTEP: int =100000, strategy: str=None) -> List[int]:
        self.current_tour, self.current_obj = self.initial_tour(strategy=strategy)
        self.best_tour, self.best_obj = list(self.current_tour), self.current_obj
        for _ in range(MAXSTEP):
            i, j = random.randint(0, self.ncity - 1), random.randint(0, self.ncity - 1)
            self.swap(i, j)
        
        return self.best_tour
    
    def solve_multi_start_two_opt(self, num_start_point: int, MAXSTEP: int =10000, strategy: str=None):
        self.best_tour = self.solve_two_opt(MAXSTEP=MAXSTEP, strategy=strategy)
        for _ in range(num_start_point - 1):
            cand = TwoOpt(self.ncity, self.D)
            cand.solve_two_opt(MAXSTEP=MAXSTEP, strategy=strategy)
            if self.best_obj > cand.best_obj:
                self.best_tour = cand.best_tour
        return self.best_tour