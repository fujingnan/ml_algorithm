import torch
import pandas as pd
import numpy as np


class SVM():
    def __init__(self):
        self.E = [0 for _ in range(self.traindata.size(0))]
        self.alpha = [0 for _ in range(self.traindata.size(0))]
        self.support_vectors = []

    def init_arg(self, traindata, labels, c, soft, iters, ep, sigma, b=0):
        self.C = c
        self.soft = soft
        self.iters = iters
        self.ep = ep
        self.traindata = torch.Tensor(traindata)
        self.labels = labels
        self.row, self.col = self.traindata.size()
        self.sigma = sigma
        self.b = b
        self.k = self.kernel()

    def kernel(self):
        """
        定义核函数，这里使用高斯核函数
        :return: 高斯核k
        """
        k = [[0] * self.row for _ in range(self.row)]
        k = np.array(k)
        for i in range(self.row):
            _x = self.traindata[i, :]
            for j in range(i, self.row):
                _z = self.traindata[j, :]
                # 参考公式7.90
                gauss = np.exp(-1 * np.dot(_x - _z, (_x - _z).T) / (2 * pow(self.sigma, 2)))
                k[i][j] = gauss
                k[j][i] = gauss
        k = torch.from_numpy(k)
        return k

    def gxi(self, i):
        g = 0
        alphas = [index for index, x in enumerate(self.alpha) if x]
        for index in alphas:
            g += self.alpha[index] * self.labels[index] * self.k[i][index]
        g += self.b
        return g

    def Ei(self, i):
        gi = self.gxi(i)
        yi = self.labels[i]
        e = gi - yi
        return e

    def anotherE(self, e1, i1):
        index = [i for i, x in enumerate(self.E) if x]
        e2_index = self.row
        if e1 >= 0:
            e2 = float('-inf')
            for j in index:
                if self.E[j] > e2:
                    e2 = self.E[j]
                    e2_index = j
        else:
            e2 = float('inf')
            for j in index:
                if self.E[j] < e2:
                    e2 = self.E[j]
                    e2_index = j
        if e2_index == self.row:
            e2_index = i1
            while e2_index == i1:
                e2_index = np.random.randint(0, self.row)
        return e2, e2_index

    def isKKT(self, i):
        gi = self.gxi(i)
        yi = self.labels[i]
        if self.alpha[i] == 0:
            return yi * gi >= 1
        elif self.alpha[i] > 0 and self.alpha[i] < self.C:
            return yi == 1
        elif self.alpha[i] == self.C:
            return yi <= 1
        return False

    def calLH(self, i, j):
        if self.labels[i] == self.labels[j]:
            l, h = max(0, self.alpha[j] - self.alpha[i]), min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            l, h = max(0, self.alpha[j] + self.alpha[i] - self.C), min(self.C, self.alpha[j] + self.alpha[i])
        return l, h

    def fit(self, traindata, labels, c, soft, sigma, b=0, iters=50, ep=10e-6):
        self.init_arg(traindata, labels, c, soft, iters, ep, sigma, b)
        early_stop = 1
        while iters > 0 and early_stop:
            early_stop = 0
            iters -= 1
            for i in range(self.row):
                if not self.isKKT(i):
                    E1 = self.Ei(i)
                    E2, j = self.anotherE(E1, i)
                    y1, y2 = self.labels[i], self.labels[j]
                    L, H = self.calLH(i, j)
                    if L == H:
                        continue
                    k11 = self.k[i][i]
                    k12 = self.k[i][j]
                    k21 = self.k[j][i]
                    k22 = self.k[j][j]
                    alphaOld_1 = self.alpha[i]
                    alphaOld_2 = self.alpha[j]
                    alphaNew_2_unc = alphaOld_2 + y2 * (E2 - E1) / (k11 - 2* k22 + k12)
                    if alphaNew_2_unc < L:
                        alphaNew_2 = L
                    elif alphaNew_2_unc > H:
                        alphaNew_2 = H
                    else:
                        alphaNew_2 = alphaNew_2_unc
                    alphaNew_1 = alphaOld_1 + y1 * y2 * (alphaOld_2 - alphaNew_2)
                    b1_new = -E1 - y1 * k11 * (alphaNew_1 - alphaOld_1) - y2 * k21 \
                            * (alphaNew_2 - alphaOld_2) + self.b
                    b2_new = -E2 - y1 * k12 * (alphaNew_1 - alphaOld_1) - y2 * k22 \
                            * (alphaNew_2 - alphaOld_2) + self.b
                    if alphaNew_1 > 0 and alphaNew_1 < self.C:
                        b_new = b1_new
                    elif alphaNew_2 > 0 and alphaNew_2 < self.C:
                        b_new = b2_new
                    else:
                        b_new = (b1_new + b2_new) / 2
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.E[i] = E1
                    self.E[j] = E2
                    self.b = b_new
                    if alphaNew_2 - alphaNew_1 > ep:
                        early_stop += 1

        for i, al in enumerate(self.alpha):
            if al > 0:
                self.support_vectors.append((i, al))
