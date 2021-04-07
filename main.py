from time import time
from math import isclose
from argparse import ArgumentParser
from utils import read, plot, calc_dist
from tsp_ip import PulpIP
from tsp_two_opt import TwoOpt
from tsp_simulated_annealing import TSPSimAnneal
from tsp_ga import TSPGA

def argparser():
    parser = ArgumentParser()
    parser.add_argument("-f", default="./ALL_tsp/ulysses16.tsp")
    return parser

def main(filename):
    name, ncity, D, coord = read(filename)

    two_opt = TwoOpt(ncity, D)
    two_opt_multi = TwoOpt(ncity, D)
    simanneal = TSPSimAnneal(ncity, D)
    ga = TSPGA(ncity, D)
    IP = PulpIP(ncity, D)

    ts = time()
    tour = two_opt.solve_two_opt(strategy="random")
    total_dist = calc_dist(tour, D)
    assert isclose(total_dist, two_opt.best_obj, abs_tol=1e-5)
    print("\nnormal", total_dist, "\ntime", time()-ts)

    ts = time()
    tour = two_opt_multi.solve_multi_start_two_opt(10)
    total_dist = calc_dist(tour, D)
    assert isclose(total_dist,  two_opt_multi.best_obj, abs_tol=1e-5)
    print("\nmulti start", total_dist, "\ntime", time()-ts)
    
    ts = time()
    tour = simanneal.solve_simulated_annealing(T=1e5)
    total_dist = calc_dist(tour, D)
    assert isclose(total_dist,  simanneal.best_obj, abs_tol=1e-5)
    print("\nsimulated annealing", total_dist, "\ntime", time()-ts)
    
    ts = time()
    tour = ga.solve()
    total_dist = calc_dist(tour, D)
    print("\nga simple", total_dist, "\ntime", time()-ts)
    
    ts = time()
    tour = IP.solve()
    total_dist = calc_dist(tour, D)
    print("\nip", total_dist, "\ntime", time()-ts)
    
    #plot(tour, coord, figname=f"./{name}_{total_dist}.png")

if __name__=="__main__":
    parser = argparser()
    args = parser.parse_args()
    main(args.f)
# ToDo
"""
+ optunaによるパラメータチューニングの実装
+ amplifyによるQAの実装
+ MTZ制約を強くした版の実装
+ 凸包による初期化
+ Tabu search
+ ant colony
+ swapをgreedyにやる
+ beam search みたいな
+ GAのバリエーション
"""