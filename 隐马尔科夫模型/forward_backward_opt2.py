"""
主题: 手写前-后向算法实现
作者: 傅景楠
作者Blog：https://blog.csdn.net/qq_36583400
参考文献: 统计学习方法-II 李航
本次代码实现验证数据引用课本《统计学习方法》中例10.2，代码结果与书中结果一致；
书中后向算法并无实例，可参照本人博客中的讲解来验证结果；
大家可参照本人博客介绍阅读代码：https://blog.csdn.net/qq_36583400/article/details/128293185
"""

import numpy as np

class FB:
    def __init__(self, pi, A, B, V):
        """
        初始化模型参数
        :param pi: 初始状态概率向量
        :param A: 已学习得到的状态转移概率矩阵，这里直接引用课本中的例子10.2
        :param B: 已学习得到的概率矩阵，这里直接引用课本中的例子10.2
        :param V: 已知的观测集合，同样使用例子中的值
        """
        self.pi = np.array(pi)
        self.A = np.array(A)
        self.B = np.array(B)
        self.V = V

    def cal_prob(self, O, opt):
        if opt == 'f':
            metrix = self.forward(O)
            # 计算P(O|λ) 公式10.17
            return sum(metrix[-1])
        elif opt == 'b':
            # 计算P(O|λ) 公式10.21
            metrix = self.backward(O)
            return sum(metrix[-1])


    def forward(self, O):
        """
        前向算法
        :param O: 已知的观测序列
        :return: P(O|λ)
        """
        row, col = len(O), self.A.shape[0]
        alpha_t_plus_1 = np.zeros((row, col), dtype=float)
        obj_index = self.V.index(O[0])
        # 初值α 公式10.15
        alpha_t_plus_1[0][:] = self.pi * self.B[:].T[obj_index]
        for t, o in enumerate(O[1:]):
            t += 1
            # 递推 公式10.16
            obj_index = self.V.index(o)
            alpha_ji = alpha_t_plus_1[t - 1][:].T @ self.A
            alpha_t_plus_1[t][:] = alpha_ji * self.B[:].T[obj_index]

        return alpha_t_plus_1

    def backward(self, O):
        """
        后向算法
        :param O: 已知的观测序列
        :return: P(O|λ)
        """
        row, col = len(O), self.A.shape[0]
        betaT = np.zeros((row+1, col), dtype=float)
        # 初值β 公式10.19
        betaT[0][:] = [1] * self.A.shape[0]
        for t, o in enumerate(O[::-1][1:]):
            t += 1
            # 反向递推 公式10.20
            obj_index = self.V.index(O[t-1])
            beta_t = self.A * self.B[:].T[obj_index] @ betaT[t-1][:].T
            betaT[t][:] = beta_t
        betaT[-1][:] = [self.pi[i] * self.B[i][self.V.index(O[0])] * betaT[-2][i] for i in range(self.A.shape[0])]
        return betaT

if __name__ == '__main__':
    from time import time
    # 课本例子10.2
    pi = [0.2, 0.4, 0.4]
    a = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ])
    b = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]
    ])
    O = ['红', '白', '红']
    f = FB(pi=pi, A=a, B=b, V=['红', '白'])
    start = time()
    resf = f.forward(O)
    resb = f.backward(O)
    print(time() - start)
    print(f.cal_prob(O, opt='b'))