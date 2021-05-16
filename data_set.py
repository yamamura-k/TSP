import torch
from torch.utils.data import Dataset
import numpy as np
import itertools
from tqdm import tqdm
from scipy.spatial.distance import cdist
from tsp_ip import PulpIP
from multiprocessing import Pool
import pickle
import os

def wrap_TSP_IP(coordinates):
    ncity = len(coordinates)

    if pow(2, ncity)*pow(ncity, 2) < 1e5:
        print("Solve with dp.")
        return tsp_dp_opt(coordinates)
        
    D = cdist(coordinates, coordinates)
    solver = PulpIP(ncity, D, MTZ_level=2)
    solution = solver.solve()
    return np.asarray(solution)

def tsp_dp_opt(coordinates):
    D = cdist(coordinates, coordinates)
    # Initial value - just distance from 0 to every other point + keep the track of edges
    A = {(frozenset([0, idx+1]), idx+1): (dist, [0, idx+1]) for idx, dist in enumerate(D[0][1:])}
    cnt = len(coordinates)
    for m in range(2, cnt):
        B = {}
        for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
            for j in S - {0}:
                # This will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
                B[(S, j)] = min([(A[(S-{j}, k)][0] + D[k][j], A[(S-{j}, k)][1] + [j])
                                 for k in S if k != 0 and k != j])
        A = B
    res = min([(A[d][0] + D[0][d[1]], A[d][1]) for d in iter(A)])
    return np.asarray(res[1])

class TSPDataset(Dataset):
    def __init__(self, data_size, seq_len, solver=wrap_TSP_IP, solve=True):
        self.data_size = data_size
        self.seq_len = seq_len
        self.solver = solver
        self.solve = solve
        self.data = self._generate_data()
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data['Coordinates'][idx]).float()
        solution = torch.from_numpy(self.data['Solutions'][idx]).long()
        sample = {'coordinate': tensor, "solution": solution}
        return sample
    
    def _generate_data(self, processes=2):
        coordinates = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit='data')

        data_dict = f"dataset_{self.data_size}_{self.seq_len}.pkl"
        if os.path.isfile(data_dict):
            with open(data_dict, "rb") as f:
                data = pickle.load(f)
        else:
            for i, _ in enumerate(data_iter):
                data_iter.set_description(f"Data points{i+1/self.data_size}")
                coordinates.append(np.random.random((self.seq_len, 2)))
            solutions_iter = tqdm(coordinates, unit='solve')
    
            if self.solve:
                with Pool(processes=processes) as pool:
                    solutions = pool.map(self.solver, solutions_iter)
                list(tqdm(solutions, total=len(coordinates)))
                """
                for i, coord in enumerate(solutions_iter):
                    solutions_iter.set_description(f"Solved {i+1}/{len(coordinates)}")
                    solutions.append(self.solver(coord))
                """
            else:
                solutions = np.zeros((len(coordinates), 2))
            data = {"Coordinates": coordinates, "Solutions": solutions}
            with open(data_dict, "wb") as f:
                pickle.dump(data, f)

        return data
    
    def _to_onehot(self, coordinates):
        vec = np.zeros((len(coordinates), self.seq_len))
        for i, v in enumerate(vec):
            v[coordinates[i]] = 1
        
        return vec