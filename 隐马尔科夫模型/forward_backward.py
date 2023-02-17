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
        self.pi = pi
        self.A = A
        self.B = B
        self.V = V

    def forward(self, O):
        """
        前向算法
        :param O: 已知的观测序列
        :return: P(O|λ)
        """
        alpha_t_plus_1 = []
        for t, o in enumerate(O):
            if t == 0:
                # 初值α 公式10.15
                alpha1 = []
                for i, p in enumerate(self.pi):
                    obj_index = self.V.index(o)
                    alpha1.append(p * self.B[i][obj_index])
                alpha_t_plus_1 = alpha1
            else:
                # 递推 公式10.16
                alpha_temp = []
                for i in range(self.A.shape[0]):
                    alpha_ji = 0.
                    # 公式10.16里中括号的内容
                    for j, a in enumerate(alpha_t_plus_1):
                        alpha_ji += (a * self.A[j][i])
                    obj_index = self.V.index(o)
                    # 公式10.16
                    alpha_temp.append(alpha_ji * self.B[i][obj_index])
                alpha_t_plus_1 = alpha_temp
        # 计算P(O|λ) 公式10.17
        P = sum(alpha_t_plus_1)
        return P

    def backward(self, O):
        """
        后向算法
        :param O: 已知的观测序列
        :return: P(O|λ)
        """
        betaT = []
        for t, o in enumerate(O[::-1]):
            if t == 0:
                # 初值β 公式10.19
                betaT = [1] * self.A.shape[0]
                continue
            else:
                # 反向递推 公式10.20
                betaT_temp = []
                for i in range(self.A.shape[0]):
                    beta_t = 0.
                    obj_index = self.V.index(O[t - 1])
                    for j, b in enumerate(betaT):
                        beta_t += (self.A[i][j] * self.B[j][obj_index] * b)
                    betaT_temp.append(beta_t)
                betaT = betaT_temp
        # 计算P(O|λ) 公式10.21
        P = sum([self.pi[i]*self.B[i][self.V.index(O[0])]*betaT[i] for i in range(self.A.shape[0])])
        return P

if __name__ == '__main__':
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
    f = FB(pi=pi, A=a, B=b, V=['红', '白'])
    print(f.backward(['红', '白', '红']))