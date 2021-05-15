import torch
from torch.utils.data import Dataset
import numpy as np
import itertools
from tqdm import tqdm
from scipy.spatial.distance import cdist
from tsp_ip import PulpIP

def wrap_TSP_IP(coordinates):
    
    ncity = len(coordinates)
    D = cdist(coordinates, coordinates)
    solver = PulpIP(ncity, D, MTZ_level=2)
    solution = solver.solve()

    return np.asarray(solution)

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
        solution = torch.from_numpy(self.data['Solutions'][idx]).long() if self.solve else None
        sample = {'coordinate': tensor, "solution": solution}
        return sample
    
    def _generate_data(self):
        coordinates = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit='data')

        for i, _ in enumerate(data_iter):
            data_iter.set_description(f"Data points{i+1/self.data_size}")
            coordinates.append(np.random.random((self.seq_len, 2)))
        solutions_iter = tqdm(coordinates, unit='solve')

        if self.solve:
            for i, coord in enumerate(solutions_iter):
                solutions_iter.set_description(f"Solved {i+1}/{len(coordinates)}")
                solutions.append(self.solver(coord))
        else:
            solutions = None
        data = {"Coordinates": coordinates, "Solutions": solutions}
        
        return data
    
    def _to_onehot(self, coordinates):
        vec = np.zeros((len(coordinates), self.seq_len))
        for i, v in enumerate(vec):
            v[coordinates[i]] = 1
        
        return vec