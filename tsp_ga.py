from tsp_two_opt import TwoOpt
from deap import algorithms, base, creator, tools
from typing import List, Tuple
import optuna
class TSPGA(TwoOpt):
    def __init__(self, ncity: int, D: List[float], indpb: float =0.05, toursize: int =3, npop: int =300) -> None:
        super().__init__(ncity, D)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()

        self.toolbox.register("random_init", self.initial_tour_ga, "random")
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.random_init)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxPartialyMatched)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=indpb)
        self.toolbox.register("select", tools.selTournament, tournsize=toursize)
        self.toolbox.register("evaluate", self._eval)

        self.pop = self.toolbox.population(n=npop)
        
    def initial_tour_ga(self, strategy: str =None) -> list:
        return self.initial_tour(strategy=strategy)[0]

    def _eval(self, tour: List[int]) -> Tuple[float]:
        return self.calc_score(tour),
    
    def solve(self, cxpb: float =0.7, mutpb: float =0.2, ngen: int =40, hof: int=1) -> object:
        if hof:
            hof = tools.HallOfFame(hof)
        algorithms.eaSimple(self.pop, self.toolbox, cxpb, mutpb, ngen, halloffame=hof, verbose=False)

        return hof[0]

    def _objective(self, trial: object) -> float:
        cxpb = trial.suggest_uniform("cxpb", 1e-3, 1.0)
        mutpb = trial.suggest_uniform("mutpb", 1e-3, 1.0)
        ngen = trial.suggest_int("ngen", 40, 100)
        return self.calc_score(self.solve(cxpb=cxpb, mutpb=mutpb, ngen=ngen))
    
    def opt_hypara(self, ntrials: int =1000, timeout: int =300) -> object:
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=ntrials, timeout=timeout)
        return study.best_trial