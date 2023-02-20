import numpy as np

class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Baum_Welch:

    def __init__(self, N, M, V):
        self.A = np.random.dirichlet(np.ones(N), size=N)  # 状态转移概率矩阵
        self.B = np.random.dirichlet(np.ones(M), size=N)  # 观测概率矩阵
        self.pi = np.array(np.random.dirichlet(np.ones(N), size=1))[0]  # 初始状态概率矩阵
        self.V = V
        self.N = N
        self.M = M

    def forward(self):
        """
        前向算法
        :param O: 已知的观测序列
        :return: alpha_{i}
        """
        row, col = len(self.O), self.A.shape[0]
        alpha_t_plus_1 = np.zeros((row, col), dtype=float)
        obj_index = self.V.index(self.O[0])
        # 初值α 公式10.15
        alpha_t_plus_1[0][:] = self.pi * self.B[:].T[obj_index]
        for t, o in enumerate(self.O[1:]):
            t += 1
            # 递推 公式10.16
            obj_index = self.V.index(o)
            alpha_ji = alpha_t_plus_1[t - 1][:].T @ self.A
            alpha_t_plus_1[t][:] = alpha_ji * self.B[:].T[obj_index]

        self.alpha = alpha_t_plus_1

    def backward(self):
        """
        后向算法
        :param O: 已知的观测序列
        :return: beta_{i}
        """
        row, col = len(self.O), self.A.shape[0]
        betaT = np.zeros((row + 1, col), dtype=float)
        # 初值β 公式10.19
        betaT[0][:] = [1] * self.A.shape[0]
        for t, o in enumerate(self.O[::-1][1:]):
            t += 1
            # 反向递推 公式10.20
            obj_index = self.V.index(self.O[t - 1])
            beta_t = self.A * self.B[:].T[obj_index] @ betaT[t - 1][:].T
            betaT[t][:] = beta_t
        betaT[-1][:] = [self.pi[i] * self.B[i][self.V.index(self.O[0])] * betaT[-2][i] for i in range(self.A.shape[0])]
        self.beta = betaT[1:]

    def gamma(self, t, i):
        numerator = self.alpha[t][i] * self.beta[t][i]
        denominator = 0.

        for j in range(self.N):
            denominator += (self.alpha[t][j] * self.beta[t][j])

        return numerator / denominator

    def ksi(self, t, i, j):
        obj_index = self.V.index(self.O[t + 1])
        numerator = self.alpha[t][i] * self.A[i][j] * self.B[j][obj_index] * self.beta[t + 1][j]
        denominator = 0.

        for i in range(self.N):
            for j in range(self.N):
                denominator += self.alpha[t][i] * self.A[i][j] * self.B[j][obj_index] * self.beta[t + 1][j]

        return numerator / denominator

    def train(self, O, n, thred=0):
        self.O = O
        self.T = len(O)
        maxIter, gap = 0, thred

        while maxIter < n:
            self.forward()
            self.backward()
            # tempA = self.A
            # tempB = self.B
            # tempPi = self.pi

            maxIter += 1
            # a_{ij}
            for i in range(self.N):
                for j in range(self.N):
                    numerator = 0.
                    denominator = 0.
                    for t in range(1, self.T-1):
                        numerator += self.ksi(t, i, j)
                        denominator += self.gamma(t, i)
                    self.A[i][j] = numerator / denominator

            # b_{i}{j}
            for j in range(self.N):
                for k in range(self.M):
                    numerator = 0.
                    denominator = 0.
                    for t in range(1, self.T):
                        if self.O[t] == self.V[k]:
                            numerator += self.gamma(t, j)
                        denominator += self.gamma(t, j)
                    self.B[j][k] = numerator / denominator

            # π_{i}
            for i in range(self.N):
                self.pi[i] = self.gamma(0, i)

        return AttrDict(
            pi=self.pi,
            A=self.A,
            B=self.B
        )

if __name__ == '__main__':
    bm = Baum_Welch(N=3, M=2, V=['红', '白'])
    O = ['红', '白', '红']
    res = bm.train(O, 100)
    print(res)