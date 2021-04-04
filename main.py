from argparse import ArgumentParser
from utils import read, plot, calc_dist
from tsp_ip import solve
from tsp_two_opt import TwoOpt

def argparser():
    parser = ArgumentParser()
    parser.add_argument("-f", default="./ALL_tsp/berlin52.tsp")
    return parser
def main(filename):
    name, ncity, D, coord = read(filename)
    two_opt = TwoOpt(ncity, D)
    tour = two_opt.solve_two_opt()
    #tour = solve(ncity, D)
    total_dist = calc_dist(tour, D)
    plot(tour, coord, figname=f"./{name}_{total_dist}.png")

if __name__=="__main__":
    parser = argparser()
    args = parser.parse_args()
    main(args.f)
