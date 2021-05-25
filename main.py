import torch
import numpy as np

from time import time
from math import isclose
from argparse import ArgumentParser
from utils import read, plot, calc_dist
from tsp_ip import PulpIP
from tsp_two_opt import TwoOpt
from tsp_simulated_annealing import TSPSimAnneal
from tsp_ga import TSPGA
from train import construct


def argparser():
    parser = ArgumentParser()
    parser.add_argument("-f", default="./ALL_tsp/ulysses16.tsp")
    # PtrNet architecture
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding size')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--num_lstms', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
    parser.add_argument('--bidir', default=True, action='store_true', help='Bidirectional')
    return parser

def test1(filename):
    name, ncity, D, coord = read(filename)

    two_opt = TwoOpt(ncity, D)
    two_opt_multi = TwoOpt(ncity, D)
    simanneal = TSPSimAnneal(ncity, D)
    ga = TSPGA(ncity, D)
    IP = PulpIP(ncity, D, MTZ_level=2)

    ts = time()
    tour = two_opt.solve_two_opt(strategy="greedy_random")
    total_dist = calc_dist(tour, D)
    assert isclose(total_dist, two_opt.best_obj, abs_tol=1e-5)
    print("\nnormal", total_dist, "\ntime", time()-ts)

    ts = time()
    tour = two_opt_multi.solve_multi_start_two_opt(10, strategy="greedy_random")
    total_dist = calc_dist(tour, D)
    assert isclose(total_dist,  two_opt_multi.best_obj, abs_tol=1e-5)
    print("\nmulti start", total_dist, "\ntime", time()-ts)

    ts = time()
    tour = simanneal.solve_simulated_annealing(T=8215972750, C=0.81, strategy="greedy_random")
    total_dist = calc_dist(tour, D)
    assert isclose(total_dist,  simanneal.best_obj, abs_tol=1e-5)
    print("\nsimulated annealing", total_dist, "\ntime", time()-ts)
    
    ts = time()
    tour = ga.solve(cxpb=0.3276646451925047, mutpb=0.6116923679473824)
    total_dist = calc_dist(tour, D)
    print("\nga simple", total_dist, "\ntime", time()-ts)
    
    ts = time()
    tour = IP.solve(solver_name="cplex")
    total_dist = calc_dist(tour, D)
    print("\nip", total_dist, "\ntime", time()-ts)
    print(*tour)
    
    #plot(tour, coord, figname=f"./{name}_{total_dist}.png")

def test2(filename):
    name, ncity, D, coord = read(filename)
    simanneal = TSPSimAnneal(ncity, D)
    best_trial = simanneal.opt_hypara()
    print("total cost", best_trial.value)
    for key, val in best_trial.params.items():
        print(key, val)

    ga = TSPGA(ncity, D)
    best_trial = ga.opt_hypara()
    print("total cost", best_trial.value)
    for key, val in best_trial.params.items():
        print(key, val)

def test3(params):
    name, ncity, D, coord = read(params.f)
    model, _, _ = construct("PtrNet", params, is_train=False)
    _input = torch.Tensor([np.asarray([x for x in coord.values()])])
    ts = time()
    _, tour = model(_input)
    tour = list(tour.detach().numpy()[0])
    total_dist = calc_dist(tour, D)
    print("\nPtrNet", total_dist, "\ntime", time()-ts)
    print(*tour)

if __name__=="__main__":
    parser = argparser()
    args = parser.parse_args()
    test1(args.f)
    # test2(args.f)
    test3(args)
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
+ multi start parallel
"""
