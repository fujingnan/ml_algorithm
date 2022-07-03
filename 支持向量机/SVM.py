import torch
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import os


class SVM():
    def __init__(self, fpath=''):
        self.fpath = fpath

    def init_arg(self, c, soft, iters, ep, sigma, b, opt, traindata, labels):
        self.support_vectors = []
        self.C = c
        self.soft = soft
        self.iters = iters
        self.ep = ep
        if self.fpath and not (traindata or labels):
            self.traindata, self.labels = self.load_data(opt)
        elif (traindata and labels):
            self.traindata = torch.Tensor(traindata)
            self.labels = labels
        else:
            print('需要传入训练数据和标签，或者传入数据集路径')
        self.row, self.col = self.traindata.size()
        self.sigma = sigma
        self.b = b
        self.k = self.kernel()
        self.alpha = [np.random.randint(0, 2) for _ in range(self.traindata.size(0))]
        self.E = [self.Ei(i) for i in range(self.traindata.size(0))]

    def load_data(self, fpath='', opt='train'):
        """
        加载数据，根据自己的数据格式做相应修改，这里主要使用mnist数据集
        :param opt: 控制是否为有标签数据训练/预测，可选train、test、predict（无标签）
        :return: data or data、label
        """
        if opt=='train':
            df = pd.read_csv(self.fpath, keep_default_na=False, header=None)
        if opt == 'test' and fpath:
            df = pd.read_csv(fpath, keep_default_na=False, header=None)
        df = df.sample(n=200)
        data = []
        labels = []
        for row in tqdm(df.iterrows(), total=len(df)):
            l = row[1].tolist()
            # 归一化特征值（可选），但是归一化后加速收敛，效果较好
            data.append([int(x)/255 for x in l[1:]])
            labels.append(1 if row[1].tolist()[0] == 0 else -1)
        return torch.Tensor(data), labels

    def kernel(self):
        """
        定义核函数，这里使用高斯核函数
        :return: 高斯核k
        """
        print('计算高斯核函数')
        k = [[0] * self.row for _ in range(self.row)]
        k = np.array(k)
        for i in tqdm(range(self.row)):
            _x = self.traindata[i, :]
            for j in range(i, self.row):
                _z = self.traindata[j, :]
                # 参考公式7.90
                gauss = np.exp(-1 * np.dot(_x - _z, (_x - _z).T) / (2 * pow(self.sigma, 2)))
                k[i][j] = gauss
                k[j][i] = gauss
        # k = torch.from_numpy(k)
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
        return float(e)

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

    def fit(self, c=5, soft=0.001, iters=10, ep=10e-6, sigma=10, b=0, opt='train', traindata='', labels=''):
        print('---开始训练---')
        self.init_arg(c, soft, iters, ep, sigma, b, opt, traindata, labels)
        early_stop = 1
        maxiter = iters
        while iters > 0 and early_stop:
            print('第 {} 轮迭代'.format(maxiter - iters + 1))
            early_stop = 0
            iters -= 1
            for i in tqdm(range(self.row)):
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
                self.support_vectors.append((i, al, self.traindata[i, :]))

    def calculateKernel(self, support_x, x):
        support_x = np.array(support_x)
        x = np.array(x)
        gauss_enum = np.dot(support_x - x, (support_x - x).T)
        gauss = np.exp(-1 * gauss_enum / (2 * pow(self.sigma, 2)))
        return gauss

    def predict(self, x):
        feature = x
        supports = []
        g = 0
        for i, al, ft in self.support_vectors:
            gauss = self.calculateKernel(ft, feature)
            g += al * self.labels[i] * gauss
            supports.append(ft)
        g += self.b
        return 1 if g > 0 else -1, supports

    def test(self, testdata, testlabel, opt=''):
        predicts = []
        sups = []
        for i in range(len(testdata)):
            x = testdata[i]
            res, sup = self.predict(x)
            predicts.append(res)
            if opt:
                sups.append(sup)
        pn, nn = 0, 0
        ppred_d, npred_d = 0, 0
        prec_d, nrec_d = dict(Counter(testlabel))[-1], dict(Counter(testlabel))[1]
        for i in range(len(predicts)):
            if predicts[i] == testlabel[i]:
                if predicts[i] == 1:
                    pn += 1
                else:
                    nn += 1
            if predicts[i] == 1:
                ppred_d += 1
            else:
                npred_d += 1
        print('正样本准确率/召回率: {}/{}\n'.format(pn / ppred_d if ppred_d else 0, pn / prec_d),
              '负样本准确率/召回率: {}/{}\n'.format(nn / npred_d if npred_d else 0, nn / nrec_d))
        return predicts if not opt else (predicts, sups)

if __name__ == '__main__':
    svm = SVM(fpath='/Users/fujingnan/PycharmProjects/ml_algorithm/transMnist/Mnist/mnist_train.csv')
    svm.fit()
    testdata, labels = svm.load_data('/Users/fujingnan/PycharmProjects/ml_algorithm/transMnist/Mnist/mnist_test.csv', 'test')
    pred = svm.test(testdata, labels)
    print(pred)
