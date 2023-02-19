import numpy as np
from forward_backward_opt2 import FB

class Baum_Welch:
    def __init__(self, O, N, M, V):
        self.O = O
        self.A = np.random.dirichlet(np.ones(N), size=N)  # 状态转移概率矩阵
        self.B = np.random.dirichlet(np.ones(M), size=N)  # 观测概率矩阵
        self.Pi = np.array(np.random.dirichlet(np.ones(N), size=1))  # 初始状态概率矩阵
        self.V = V
        self.N = N
        self.M = M


    def gamma(self, t, i):
        fb = FB(pi=self.Pi, A=self.A, B=self.B, V=self.V)
        self.alpha = FB.forward(self.O)
        self.beta = FB.backword(self.O)
        numerator = self.alpha[t][i] * self.beta[t][i]
        denuminator = 0.

        for j in range(self.N):
            denuminator += (self.alpha[t][j] * self.beta[t][j])

        return numerator / denuminator

    def ksi(self, t, i, j):
        pass