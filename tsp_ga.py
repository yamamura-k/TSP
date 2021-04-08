from tsp_two_opt import TwoOpt
from deap import algorithms, base, creator, tools
from typing import List, Tuple
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