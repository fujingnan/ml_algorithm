"""
主题: 手写Baum welch算法实现
作者: 傅景楠
作者Blog：https://blog.csdn.net/qq_36583400
参考文献: 统计学习方法-II 李航
本次代码实现验证数据引用课本《统计学习方法》中例10.2的观测，由于目前还未实现预测算法，因此未能验证；
大家可参照本人博客介绍阅读代码：https://blog.csdn.net/qq_36583400/article/details/128575653
代码实现严格按照书中的公式，方便大家理解。
"""
import random
import numpy as np

random.seed(1)  # 好像不起租用


class AttrDict(dict):
    # 一个小trick，将结果返回成一个字典格式
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Baum_Welch:

    def __init__(self, N, M, V):
        self.A = np.random.dirichlet(np.ones(N), size=N)  # 状态转移概率矩阵
        self.B = np.random.dirichlet(np.ones(M), size=N)  # 观测概率矩阵
        self.pi = np.array(np.random.dirichlet(np.ones(N), size=1))[0]  # 初始状态概率矩阵
        self.V = V # 所有可能的观测
        self.N = N # 所有可能的状态长度
        self.M = M # 所有可能的观测长度

    def forward(self):
        """
        前向算法
        :param O: 已知的观测序列
        :return: alpha_{i}
        """
        row, col = len(self.O), self.A.shape[0]
        alpha_t_plus_1 = np.zeros((row, col))
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
        betaT = np.zeros((row + 1, col))
        # 初值β 公式10.19
        betaT[0][:] = [1] * self.A.shape[0]
        for t, o in enumerate(self.O[::-1][1:]):
            t += 1
            # 反向递推 公式10.20
            obj_index = self.V.index(self.O[t - 1])
            beta_t = self.A * self.B[:].T[obj_index] @ betaT[t - 1][:].T
            betaT[t][:] = beta_t
        # betaT[-1][:] = [self.pi[i] * self.B[i][self.V.index(self.O[0])] * betaT[-2][i] for i in range(self.A.shape[0])]
        self.beta = betaT[:-1][::-1]

    def gamma(self, t, i):
        """
        根据课本公式【10.24】计算γ
        :param t: 当前时间点
        :param i: 当前状态节点
        :return: γ值
        """
        numerator = self.alpha[t][i] * self.beta[t][i]
        denominator = 0.

        for j in range(self.N):
            denominator += (self.alpha[t][j] * self.beta[t][j])

        return numerator / denominator

    def ksi(self, t, i, j):
        """
        根据公式【10.26】计算 ξ
        :param t: 当前时间点
        :param i: 当前状态节点
        :param j: 同i
        :return:
        """
        obj_index = self.V.index(self.O[t + 1])
        numerator = self.alpha[t][i] * self.A[i][j] * self.B[j][obj_index] * self.beta[t + 1][j]
        denominator = 0.

        for i in range(self.N):
            for j in range(self.N):
                denominator += self.alpha[t][i] * self.A[i][j] * self.B[j][obj_index] * self.beta[t + 1][j]

        return numerator / denominator

    def train(self, O, n):
        """
        根据算法【10.4】训练模型
        :param O: 已知观测序列
        :param n: 最大迭代步长
        :return: 模型参数λ=(π，A，B)
        """
        self.O = O
        self.T = len(O)
        maxIter = 0

        while maxIter < n:
            tempA = np.zeros((self.N, self.N))
            tempB = np.zeros((self.N, self.M))
            tempPi = np.array([0.] * self.N)

            # 根据前向算法和后向算法得到α和β
            self.forward()
            self.backward()

            maxIter += 1
            # a_{ij}，公式【10.39】
            for i in range(self.N):
                for j in range(self.N):
                    numerator = 0.
                    denominator = 0.
                    for t in range(self.T - 1):
                        numerator += self.ksi(t, i, j)
                        denominator += self.gamma(t, i)
                    tempA[i][j] = numerator / denominator

            # b_{i}{j}，公式【10.40】
            for j in range(self.N):
                for k in range(self.M):
                    numerator = 0.
                    denominator = 0.
                    for t in range(self.T):
                        if self.O[t] == self.V[k]:
                            numerator += self.gamma(t, j)
                        denominator += self.gamma(t, j)
                    tempB[j][k] = numerator / denominator

            # π_{i}，公式【10.41】
            for i in range(self.N):
                tempPi[i] = self.gamma(0, i)
            # 更新
            self.A = tempA
            self.B = tempB
            self.pi = tempPi

        return AttrDict(
            pi=self.pi,
            A=self.A,
            B=self.B
        )


if __name__ == '__main__':
    bm = Baum_Welch(N=3, M=2, V=['红', '白'])
    O = ['红', '白', '红']
    res = bm.train(O, 3)
    print(res.pi)
    print(res.A)
    print(res.B)
