from scipy.spatial.distance import cdist
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm
from math import sqrt
import matplotlib.pyplot as plt
from geopy.distance import geodesic

def read(filename):
    coord = dict()
    D = []
    _type = None
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip()
            line = line.split(":")
            for l in range(len(line)):
                line[l] = line[l].replace(" ", "")
            if line[0] == "NAME":
                name = line[1]
            elif line[0] == "DIMENSION":
                ncity = int(line[1])
            elif line[0] == "EDGE_WEIGHT_SECTION":
                D = [ _line.split() for _line in f]
                D = list(list(map(float, x))for x in D[:-1])
            elif line[0] == "EDGE_WEIGHT_TYPE":
                _type = line[1]
            elif line[0] == "NODE_COORD_SECTION":
                for _line in f:
                    _line = _line.rstrip().split()

                    if _line[0] != "EOF":
                        _line = list(map(float, _line))
                        coord[_line[0] - 1] = _line[1:]
                    else:
                        break
    if coord and not D:
        XY = np.array(list(coord.values()))
        if _type == "GEO":
            for u in XY:
                tmp = []
                for v in XY:
                    tmp.append(geodesic(u, v).m)
                D.append(tmp)
        else:
            D = cdist(XY, XY)

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
    x = [coord[i][0] for i in tour] + [coord[tour[0]][0]]
    y = [coord[i][1] for i in tour] + [coord[tour[0]][1]]
    plt.plot(x, y, "-o")
    if figname:
        plt.savefig(figname)
    else:
        plt.show()

def calc_dist(tour, D):
    return sum(D[i][j] for i, j in zip(tour, tour[1:]+tour[:1]))

class TSPDataset(Dataset):
    
    def __init__(self, dataset_fnames=None, train=False, size=50, num_samples=1000000, random_seed=1111):
        super(TSPDataset, self).__init__()
        
        torch.manual_seed(random_seed)

        self.data_set = []
        if not train:
            for fname in tqdm(dataset_fnames):
                name, ncity, D, coord = read(fname)
                x = np.array([tmp for tmp in cood.values()], dtype=np.float32).reshape([-1, 2]).T
                self.data_set.append(x)
        else:
            for l in tqdm(range(num_samples)):
                x = torch.FloatTensor(2, size).uniform_(0, 1)
                self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]

def reward(sample_solution):
    batch_size = sample_solution[0].size(0)
    n = len(sample_solution)
    tour_len = Variable(torch.zeros([batch_size]))

    for i in range(n-1):
        tour_len += torch.norm(sample_solution[i] - sample_solution[i+1], dim=1)
    
    tour_len += torch.norm(sample_solution[n-1] - sample_solution[0], dim=1)
    return tour_len

if __name__=="__main__":
    name, ncity, D, coord = read("./ALL_tsp/ulysses16.tsp")
    tour = list(range(1, ncity + 1))
    plot(tour, coord, figname=f"./{name}.png")
