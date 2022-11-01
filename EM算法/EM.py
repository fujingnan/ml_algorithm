"""
主题: 手写EM-GMM算法实现
作者: 傅景楠
作者Blog：https://blog.csdn.net/qq_36583400
参考文献: 统计学习方法-II 李航
本次数据集为自己构造的混合高斯分布数据集，本次实验取两个高斯分布
大家可参照本人博客介绍阅读代码：https://blog.csdn.net/qq_36583400/article/details/127047093
"""
import numpy as np
import math
import random

class EM:
    def __init__(self, phi_1, phi_2, miu1, miu2, sigma1, sigma2, dataSize):
        """
        参数初始化
        :param phi_1: 隐变量取Gauss1的概率
        :param phi_2: 隐变量取Gauss2的概率
        :param miu1: Gauss1的伪均值
        :param miu2: Gauss2的伪均值
        :param sigma1: Gauss1的方差
        :param sigma2: Gauss2的方差
        :param dataSize: 样本数据长度
        """
        self.phi_1 = phi_1
        self.phi_2 = phi_2
        self.miu1 = miu1
        self.miu2 = miu2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.dataSize = dataSize

    def creat_gauss_dist(self):
        """
        构造一个高斯混合样本集
        :return:
        """
        data1 = np.random.normal(self.miu1, self.sigma1, int(self.dataSize * self.phi_1))
        data2 = np.random.normal(self.miu2, self.sigma2, int(self.dataSize * self.phi_2))
        dataset = []
        dataset.extend(data1)
        dataset.extend(data2)
        random.shuffle(dataset)

        return dataset

    def calculate_gauss(self, dataset, miu, sigma):
        """
        计算高斯核函数
        :param miu: 高斯核伪均值
        :param sigma: 高斯核方差
        :return: 高斯分布概率值
        """
        gauss = (1 / (math.sqrt(2 * math.pi) * sigma)) * \
                 np.exp(-1 * (dataset - miu) * (dataset - miu) / (2 * sigma ** 2))

        return gauss

    def E_step(self, dataset, phi_1, phi_2, miu1, miu2, sigma1, sigma2):
        """
        E步：
        计算Q函数，计算方法雷同《统计学习方法》p165 算法9.2 E步
        :return: Q_k(z), k=1, 2
        """

        q1_numerator = phi_1 * self.calculate_gauss(dataset, miu1, sigma1)
        q2_numerator = phi_2 * self.calculate_gauss(dataset, miu2, sigma2)

        q_denominator = q1_numerator + q2_numerator

        q1 = q1_numerator / q_denominator
        q2 = q2_numerator / q_denominator

        return q1, q2

    def M_step(self, dataset, miu1, miu2, q1, q2):
        """
        M步：
        计算参数的最大似然估计，计算方法雷同《统计学习方法》p165 算法9.2 M步
        """

        nk1 = np.sum(q1)
        nk2 = np.sum(q2)

        phi_new_1 = np.sum(q1) / len(q1)
        phi_new_2 = np.sum(q2) / len(q2)

        miu_new_1 = np.dot(q1, dataset) / nk1
        miu_new_2 = np.dot(q2, dataset) / nk2

        sigma_new_1 = math.sqrt(np.dot(q1, (dataset - miu1) ** 2) / nk1)
        sigma_new_2 = math.sqrt(np.dot(q2, (dataset - miu2) ** 2) / nk2)

        return miu_new_1, miu_new_2, sigma_new_1, sigma_new_2, phi_new_1, phi_new_2

    def train(self):
        dataset = self.creat_gauss_dist()
        dataset = np.array(dataset)
        max_iter = 10000
        step = 0

        phi_1 = self.phi_1
        phi_2 = self.phi_2

        miu1 = self.miu1
        miu2 = self.miu2

        sigma1 = self.sigma1
        sigma2 = self.sigma2
        while step < max_iter:
            q1, q2 = self.E_step(dataset, phi_1=phi_1, phi_2=phi_2, miu1=miu1, miu2=miu2, sigma1=sigma1, sigma2=sigma2)
            miu1, miu2, sigma1, sigma2, phi_1, phi_2 = self.M_step(dataset, miu1, miu2, q1, q2)
            step += 1

        return miu1, miu2, sigma1, sigma2, phi_1, phi_2

if __name__ == '__main__':
    phi_1 = 0.3
    phi_2 = 0.7
    miu1 = -0.2
    miu2 = 0.5
    sigma1 = 0.9
    sigma2 = 1
    print('phi_1:%.1f, miu1:%.1f, sigma1:%.1f, phi_2:%.1f, miu2:%.1f, sigma2:%.1f' % (
        phi_1, miu1, sigma1, phi_2, miu2, sigma2
    ))
    em = EM(
        phi_1=0.3,
        phi_2=0.7,
        miu1=-0.2,
        miu2=0.5,
        sigma1=0.9,
        sigma2=1,
        dataSize=1000
    )
    miu1, miu2, sigma1, sigma2, phi_1, phi_2 = em.train()
    print('phi_1:%.1f, miu1:%.1f, sigma1:%.1f, phi_2:%.1f, miu2:%.1f, sigma2:%.1f' % (
        phi_1, miu1, sigma1, phi_2, miu2, sigma2
    ))