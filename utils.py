from scipy.spatial.distance import cdist
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def read(filename):
    coord = dict()
    D = []
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip()
            line = line.split(":")
            if "NAME" in line[0]:
                name = line[1]
            elif line[0] == "DIMENSION":
                ncity = int(line[1])
            elif line[0] == "EDGE_WEIGHT_SECTION":
                D = [ _line.split() for _line in f]
                D = list(list(map(float, x))for x in D[:-1])
            elif "EDGE_WEIGHT_TYPE" in line[0] and "EUC_2D" not in line[1]:
                print("Cannot use this problem file.")
                return None
            elif line[0] == "NODE_COORD_SECTION":
                for _line in f:
                    if _line[:3] != "EOF":
                        _line = list(map(float, _line.split()))
                        coord[_line[0]] = _line[1:]
                    else:
                        break
    if coord and not D:
        D = cdist(coord.values(), coord.values())
        """
        for v in coord.values():
            tmp = []
            for u in coord.values():
                tmp.append(sqrt(pow(u[0] - v[0], 2) + pow(u[1] - v[1], 2)))
            D.append(tmp)
        """
    return name, ncity, D, coord

def plot(tour, coord, figname="./tmp.png"):
    fig = plt.figure()
    x = [coord[i][0] for i in tour]
    y = [coord[i][1] for i in tour]
    plt.plot(x, y, "-o")
    plt.savefig(figname)

def calc_dist(tour, D):
    return sum(D[i][j] for i, j in zip(tour, tour[1:]+tour[:1]))

if __name__=="__main__":
    name, ncity, D, coord = read("./ALL_tsp/berlin52.tsp")
    tour = list(range(1, ncity + 1))
    plot(tour, coord, figname=f"./{name}.png")