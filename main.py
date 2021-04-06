from argparse import ArgumentParser
from utils import read, plot, calc_dist
from tsp_ip import PulpIP
from tsp_two_opt import TwoOpt
from tsp_simulated_annealing import TSPSimAnneal

def argparser():
    parser = ArgumentParser()
    parser.add_argument("-f", default="./ALL_tsp/ulysses16.tsp")
    return parser
def main(filename):
    name, ncity, D, coord = read(filename)

    two_opt = TwoOpt(ncity, D)
    tour = two_opt.solve_two_opt()
    total_dist = calc_dist(tour, D)
    print("normal", total_dist)

    two_opt = TwoOpt(ncity, D)
    tour = two_opt.solve_multi_start_two_opt(10)
    total_dist = calc_dist(tour, D)
    print("multi start", total_dist)

    two_opt = TSPSimAnneal(ncity, D)
    tour = two_opt.solve_simulated_annealing()
    total_dist = calc_dist(tour, D)
    print("simulated annealing", total_dist)
    
    IP = PulpIP(ncity, D)
    tour = IP.solve()
    total_dist = calc_dist(tour, D)
    print("ip", total_dist)
    
    #plot(tour, coord, figname=f"./{name}_{total_dist}.png")

if __name__=="__main__":
    parser = argparser()
    args = parser.parse_args()
    main(args.f)
