import random
import numpy as np
from typing import List, Tuple
from tsp_two_opt import TwoOpt

class TSPSimAnneal(TwoOpt):
    def __init__(self, ncity: int, D: List[float]) -> None:
        super().__init__(ncity, D)
    
    def _swap_prob(self, i: int, j: int, T: float) -> float:
        _swap_cost = self.swap_cost(i, j)
        return min(1., np.exp(-(self.current_obj + _swap_cost - self.best_obj)/T)), _swap_cost

    def swap(self, i: int, j: int, T: float):
        _swap_prob, _swap_cost = self._swap_prob(i, j, T)
        if random.random() < _swap_prob:
            self.current_tour[i], self.current_tour[j] = self.current_tour[j], self.current_tour[i]
            self.current_obj += _swap_cost
    
    def solve_simulated_annealing(self, T=1e5, C=0.995, MAXSTEP: int =100, strategy: str=None, _type: str ="inhom") -> List[int]:
        self.current_tour, self.current_obj = self.initial_tour(strategy=strategy)
        self.best_tour, self.best_obj = list(self.current_tour), self.current_obj

        # homogeneous algorithm
        # 収束が遅い
        if _type == "hom":
            while T > 100:
                for _ in range(MAXSTEP):
                    i, j = random.randint(0, self.ncity - 1), random.randint(0, self.ncity - 1)
                    self.swap(i, j, T)
                    if self.current_obj < self.best_obj:
                        self.best_tour = list(self.current_tour)
                        self.best_obj = self.current_obj
                T *= C
                
        # inhomogeneous algorithm
        elif _type == "inhom":
            while T > 100:
                i, j = random.randint(0, self.ncity - 1), random.randint(0, self.ncity - 1)
                self.swap(i, j, T)
                if self.current_obj < self.best_obj:
                    self.best_tour = list(self.current_tour)
                    self.best_obj = self.current_obj
                T *= C
        else:
            raise NotImplementedError
        return self.best_tour