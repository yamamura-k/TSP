import random
class TwoOpt:
    def __init__(self, ncity, D):
        self.ncity = ncity
        self.D = D
        self.current_tour = None
        self.best_tour = None

    def initial_tour(self, strategy=None):
        if strategy:
            raise NotImplementedError
        return list(range(1, self.ncity + 1))

    def swap_cost(self, i, j):
        if i > j:
            i, j = j, i
        i_now = self.current_tour[i] - 1
        i_prev = self.current_tour[i - 1] - 1
        i_next = self.current_tour[(i + 1)%self.ncity] - 1

        j_now = self.current_tour[j] - 1
        j_prev = self.current_tour[j - 1] - 1
        j_next = self.current_tour[(j + 1)%self.ncity] - 1
        
        current_cost = self.D[i_prev][i_now] + self.D[i_now][i_next] + self.D[j_prev][j_now] + self.D[j_now][j_next]
        new_cost = self.D[i_prev][j_now] + self.D[j_now][i_next] + self.D[j_prev][i_now] + self.D[i_now][j_next]
        return new_cost - current_cost
    
    def swap(self, i, j):
        if self.swap_cost(i, j) < 0:
            self.current_tour[i], self.current_tour[j] = self.current_tour[j], self.current_tour[i]
            self.best_tour = list(self.current_tour)
    
    def solve_two_opt(self, MAXSTEP=1000, strategy=None):
        self.current_tour = self.initial_tour(strategy=strategy)

        for _ in range(MAXSTEP):
            i, j = random.randint(0, self.ncity - 1), random.randint(0, self.ncity - 1)
            self.swap(i, j)
        
        return self.best_tour