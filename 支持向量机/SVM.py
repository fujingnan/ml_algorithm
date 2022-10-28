"""
主题: 手写SVM二分类实现
作者: 傅景楠
参考文献: 统计学习方法-II 李航
本次数据集使用mnist数据集，本实验取前1000条数据，默认参数下测试，正负样本准召率：
- 正样本准确率/召回率: 0.9/0.8470588235294118
- 负样本准确率/召回率: 0.9858695652173913/0.9912568306010929
"""

import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import math


class SVM():
    def __init__(self, fpath=''):
        self.fpath = fpath

    def init_arg(self, c, soft, iters, ep, sigma, b, opt, traindata, labels, isSoft):
        """
        参数初始化
        :param c: 惩罚参数，也可用作松弛变量，当用作松弛变量时，可以忽略soft；
                  实验对比发现，如果作为惩罚参数，一般C取值大于200效果会好些，
                  具体还需要分析数据特征进行调整，此处默认取值200
        :param soft: 松弛变量，如果C作为惩罚参数，该变量不可忽略，默认取值0.001
        :param iters: 迭代次数
        :param ep: alpha2变量的改变误差，当误差小于该值时，可以认为没有迭代的必要了
        :param sigma: 高斯核函数中的参数
        :param b: 偏置项
        :param opt: 操作标志，可选train、test
        :param traindata: 训练样本
        :param labels: 标签
        :param support_vectors: 存储支持向量，实际存储时，元素为tuple -> (支持向量索引, 索引对应的alpha值, 支持向量)
        :return: 返回类的成员变量
        """
        self.use_soft = 0
        if isSoft:
            self.use_soft = soft
        self.support_vectors = []
        self.C = c
        self.iters = iters
        self.ep = ep
        if self.fpath and not (traindata or labels):
            self.traindata, self.labels = self.load_data(opt)
        elif (traindata and labels):
            self.traindata = np.array(traindata)
            self.labels = labels
        else:
            print('需要传入训练数据和标签，或者传入数据集路径')
        self.labels = np.array(self.labels).T
        self.row, self.col = np.shape(self.traindata)
        self.sigma = sigma
        self.b = b
        self.k = self.kernel()
        self.alpha = [float(np.random.randint(0, 1)) for _ in range(self.traindata.shape[0])]
        self.E = [self.Ei(i) for i in range(self.traindata.shape[0])]
        # self.alpha = [0 for _ in range(self.traindata.shape[0])]
        # self.E = [self.Ei(i) for i in range(self.traindata.shape[0])]

    def load_data(self, fpath='', opt='train'):
        """
        加载数据，根据自己的数据格式做相应修改，这里主要使用mnist数据集
        :param opt: 控制是否为有标签数据训练/预测，可选train、test
        :return: data、label
        """
        if opt=='train':
            df = pd.read_csv(self.fpath, keep_default_na=False, header=None)
        if opt == 'test' and fpath:
            df = pd.read_csv(fpath, keep_default_na=False, header=None)
        df = df[:1000]
        data = []
        labels = []
        for row in tqdm(df.iterrows(), total=len(df)):
            l = row[1].tolist()
            # 归一化特征值（可选），但是归一化后加速收敛，效果较好
            data.append([int(x)/255 for x in l[1:]])
            labels.append(1 if row[1].tolist()[0] == 0 else -1)
        return np.array(data), labels

    def kernel(self):
        """
        定义核函数，这里使用高斯核函数；注意如果数据量较大，使用高斯核函数计算过程非常漫长，可考虑其它核函数，如线性核函数等
        :return: 高斯核k
        """
        print('计算高斯核函数')
        k = [[0] * self.row for _ in range(self.row)]
        for i in tqdm(range(self.row)):
            _x = self.traindata[i, :]
            for j in range(i, self.row):
                _z = self.traindata[j, :]
                # 参考公式7.90
                gauss = np.exp(-1 * np.dot(_x - _z, (_x - _z).T) / (2 * pow(self.sigma, 2)))
                k[i][j] = gauss
                k[j][i] = gauss
        return k

    def gxi(self, i):
        """
        分类决策函数的计算，即预测值的计算，参考公式 7.104
        :param i: 已知量的索引
        :return: 预测值
        """
        g = 0
        alphas = [index for index, x in enumerate(self.alpha) if x]
        for index in alphas:
            g += self.alpha[index] * self.labels[index] * self.k[i][index]
        g += self.b
        return g

    def Ei(self, i):
        """
        误差值的计算，也用于更新误差值，参考公式 7.105
        :param i:
        :return:
        """
        gi = self.gxi(i)
        yi = self.labels[i]
        e = gi - yi
        return float(e)

    def anotherE(self, e1, i1):
        """
        计算约束方向上alpha2_new值对应的E，参考p147中'第二个变量的选择'
        :param e1: 每轮更新中第一个基准E1，基于此来选择第二个E2
        :param i1: E1的位置索引
        :return: E2、E2位置索引
        """
        index = [i for i, x in enumerate(self.E) if x]
        e2_index = self.row
        if e1 >= 0:
            e2 = float('inf')
            for j in index:
                if self.E[j] < e2:
                    e2 = self.E[j]
                    e2_index = j
        else:
            e2 = float('-inf')
            for j in index:
                if self.E[j] > e2:
                    e2 = self.E[j]
                    e2_index = j
        if e2_index == self.row:
            e2_index = i1
            while e2_index == i1:
                e2_index = np.random.randint(0, self.row-1)
            e2 = self.Ei(e2_index)
        return e2, e2_index

    def isKKT(self, i):
        """
        kkt条件判断，参考7.111-7.113，需要注意的是当C作为惩罚参数时，kkt条件中需要考虑松弛变量，大体思路就是：
        原yi*gi的不等式条件右边加上-soft，等价于alpha的不等式条件在同样加上松弛变量后不等式符号的变化
        :param i: 当前alpha对应的下标
        :return:
        """
        gi = self.gxi(i)
        yi = self.labels[i]
        if not self.use_soft:
            if self.alpha[i] == 0:
                return yi * gi >= 1
            elif self.alpha[i] > 0 and self.alpha[i] < self.C:
                return yi == 1
            elif self.alpha[i] == self.C:
                return yi <= 1
            return False
        else:
            if (math.fabs(self.alpha[i]) < self.use_soft) and (yi * gi >= 1):
                return True
            # 参考7.113
            elif (math.fabs(self.alpha[i] - self.C) < self.use_soft) and (yi * gi <= 1):
                return True
            # 参考7.112
            elif (self.alpha[i] > -self.use_soft) and (self.alpha[i] < (self.C + self.use_soft)) \
                    and (math.fabs(yi * gi - 1) < self.use_soft):
                return True
            return False

    def calLH(self, i, j):
        """
        L、H上下界的判断，参考p144
        :param i:
        :param j:
        :return:
        """
        if self.labels[i] != self.labels[j]:
            l, h = max(0, self.alpha[j] - self.alpha[i]), min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            l, h = max(0, self.alpha[j] + self.alpha[i] - self.C), min(self.C, self.alpha[j] + self.alpha[i])
        return l, h

    def fit(self, c=200, soft=0.001, iters=100, ep=10e-6, sigma=10, b=0, opt='train', traindata='', labels='', isSoft='True'):
        """
        smo过程：
        1. 先选取两个待更新alpha1和alpha2，其中，alpha1需不满足kkt条件
        2. 根据根据选取的alpha1，计算对应位置的误差值E1
        3. 根据E1和对应的下标i计算出E2和对应的下标j
        4. 计算alphai的上下界L和H
        5. 获取计算好的核函数中的k11, k12, k21, k22
        6. 计算未剪辑的alpha2', 并根据上下界判断条件剪辑alpha2'
        7. 分别计算alpha1'、b1'、b2'
        8. 由b1'、b2'计算出b'
        9. 计算更新后的E1'、E2'
        10. 分别更新b、alpha1、alpha2、E1、E2为b'、alpha1'、alpha2'、E1'、E2'
        11. 循环上述过程
        """
        # 参数初始化
        self.init_arg(c, soft, iters, ep, sigma, b, opt, traindata, labels, isSoft)
        print('---开始训练---')
        # 早停条件
        early_stop = 1
        maxiter = iters
        while iters > 0 and early_stop:
            print('第 {}/{} 轮迭代'.format(maxiter - iters + 1, maxiter))
            early_stop = 0
            iters -= 1
            for i in range(self.row):
                # 只取不满足kkt条件的位置对应的变量
                if not self.isKKT(i):
                    # 先计算E1，再由E1和对应的下标确定E2和其对应的下标
                    E1 = self.Ei(i)
                    E2, j = self.anotherE(E1, i)
                    y1, y2 = self.labels[i], self.labels[j]
                    L, H = self.calLH(i, j)
                    # 如果上下界相等，就没有更新的必要了
                    if L == H:
                        continue
                    k11 = self.k[i][i]
                    k12 = self.k[i][j]
                    k21 = self.k[j][i]
                    k22 = self.k[j][j]
                    # smo算法，每次取两个变量alpha1和alpha2计算
                    alphaOld_1 = self.alpha[i]
                    alphaOld_2 = self.alpha[j]
                    # 参考公式 7.106-7.107
                    eta = k11 - 2 * k12 + k22
                    alphaNew_2_unc = alphaOld_2 + y2 * (E1 - E2) / eta
                    # alpha_new2的剪辑，参考 7.108
                    if alphaNew_2_unc < L:
                        alphaNew_2 = L
                    elif alphaNew_2_unc > H:
                        alphaNew_2 = H
                    else:
                        alphaNew_2 = alphaNew_2_unc
                    # 计算新的alpha1->alphaNew_1，参考公式 7.109
                    alphaNew_1 = alphaOld_1 + y1 * y2 * (alphaOld_2 - alphaNew_2)
                    # 计算新的b1_new、b2_new，参考公式 7.115-7.116
                    b1_new = -E1 - y1 * k11 * (alphaNew_1 - alphaOld_1) - y2 * k21 \
                            * (alphaNew_2 - alphaOld_2) + self.b
                    b2_new = -E2 - y1 * k12 * (alphaNew_1 - alphaOld_1) - y2 * k22 \
                            * (alphaNew_2 - alphaOld_2) + self.b
                    # 确定需要更新的b的值，参考p148
                    if alphaNew_1 > 0 and alphaNew_1 < self.C:
                        b_new = b1_new
                    elif alphaNew_2 > 0 and alphaNew_2 < self.C:
                        b_new = b2_new
                    else:
                        b_new = (b1_new + b2_new) / 2
                    # 更新各项参数，注意b的更新一定要放在其它变量更新的前面，因为其他变量的更新是基于更新后的b再次计算的
                    self.b = b_new
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.E[i] = self.Ei(i)
                    self.E[j] = self.Ei(j)
                    # 如果alpha2的变化量过小，则无需迭代了，该操作可以没有
                    if math.fabs(alphaNew_2 - alphaNew_1) > ep:
                        early_stop += 1
        # 参数全部更新完毕后，记录支持向量
        for i, al in enumerate(self.alpha):
            if al > 0:
                self.support_vectors.append((i, al, self.traindata[i, :]))

    def calculateKernel(self, support_x, x):
        # 做预测的时候，需要根据计算好的支持向量和当前需要预测的向量再算一次核函数
        support_x = np.array(support_x)
        x = np.array(x)
        gauss_enum = np.dot(support_x - x, (support_x - x).T)
        gauss = np.exp(-1 * gauss_enum / (2 * pow(self.sigma, 2)))
        return gauss

    def predict(self, x):
        """
        预测函数(无标签)
        :param x: 当前需要预测的样本
        :return: 预测值g，参考公式 7.104
        """
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
        """
        测试函数，带标签，需要用到预测函数predict
        :param testdata: 测试样本
        :param testlabel: 测试标签
        :param opt: 是否需要打印出支持向量，可作为特征重要度分析参考
        :return: 预测结果
        """
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
        prec_d, nrec_d = dict(Counter(testlabel))[1], dict(Counter(testlabel))[-1]
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
