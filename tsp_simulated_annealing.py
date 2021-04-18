import random
import optuna
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
    
    def solve_simulated_annealing(self, T: float =1e5, T_final: float =1e2, C: float =0.995, MAXSTEP: int =100, strategy: str=None, _type: str ="inhom") -> List[int]:
        self.current_tour, self.current_obj = self.initial_tour(strategy=strategy)
        self.best_tour, self.best_obj = list(self.current_tour), self.current_obj

        # homogeneous algorithm
        # 収束が遅い
        if _type == "hom":
            while T > T_final:
                for _ in range(MAXSTEP):
                    i, j = random.randint(0, self.ncity - 1), random.randint(0, self.ncity - 1)
                    self.swap(i, j, T)
                    if self.current_obj < self.best_obj:
                        self.best_tour = list(self.current_tour)
                        self.best_obj = self.current_obj
                T *= C
                
        # inhomogeneous algorithm
        elif _type == "inhom":
            while T > T_final:
                i, j = random.randint(0, self.ncity - 1), random.randint(0, self.ncity - 1)
                self.swap(i, j, T)
                if self.current_obj < self.best_obj:
                    self.best_tour = list(self.current_tour)
                    self.best_obj = self.current_obj
                T *= C
        else:
            raise NotImplementedError
        return self.best_tour

    def _objective(self, trial: object) -> float:
        T = trial.suggest_discrete_uniform("T", 1e3, 1e10, 10)
        T_final = trial.suggest_uniform("T_final", 0, 1e3)
        C = trial.suggest_uniform("C", 0.5, 1.0)
        strategy = trial.suggest_categorical("strategy", ["random", "greedy", "greedy_random", ""])
        self.solve_simulated_annealing(T=T, T_final=T_final, C=C, strategy=strategy, _type="inhom")
        return self.best_obj
    
    def opt_hypara(self, ntrials: int =1000, timeout: int =300) -> object:
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=ntrials, timeout=timeout)
        return study.best_trial