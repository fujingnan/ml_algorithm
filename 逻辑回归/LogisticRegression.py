"""
逻辑回归手写实现-torch版
作者：傅景楠
参考文献：统计学习方法-II By：李航

本次数据集使用mnist数据集，默认参数下测试，正负样本准召率：
- 正样本准确率/召回率: 0.9590163934426229/0.9551020408163265
- 负样本准确率/召回率: 0.9951241134751773/0.9955654101995566
"""
import torch
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm


class LogisticRgression():
    def __init__(self, lr=0.01, epoch=3, fpath=''):
        self.lr = lr
        self.epoch = epoch
        self.fpath = fpath

    def load_data(self, opt='train'):
        """
        加载数据，根据自己的数据格式做相应修改，这里主要使用mnist数据集
        :param opt: 控制是否为有标签数据训练/预测，可选train、test、predict（无标签）
        :return: data or data、label
        """
        df = pd.read_csv(self.fpath, keep_default_na=False, header=None)
        data = []
        labels = []
        for row in tqdm(df.iterrows(), total=len(df)):
            l = row[1].tolist()
            if opt == 'train' or opt == 'test':
                # 归一化特征值（可选），但是归一化后加速收敛，效果较好
                data.append([int(x)/255 for x in l[1:]] + [1])
                labels.append(1 if row[1].tolist()[0] == 0 else 0)
            else:
                data.append([int(x)/255 for x in l] + [1])
        return torch.Tensor(data), labels if opt == 'train' or opt == 'test' else torch.Tensor(data)

    def logistic_regression(self, x, w, label):
        """
        逻辑回归计算
        :param x: 训练样本，[1, hidden_size]
        :param w: 权重，float
        :param label: 标签，int
        :return: w
        """
        x = torch.transpose(x, 0, -1)
        w_x = torch.matmul(w, x)
        y_x = label * x
        # 根据统计学习方法6.1.3公式对w求偏导后的结果
        logic_func = y_x - np.exp(w_x)*x / (1 + np.exp(w_x))
        # 特别注意，由于是求极大似然值，而不是loss，因此这里应当使用梯度上升而非梯度下降
        w += self.lr * logic_func
        return w

    def train(self):
        """
        训练过程
        :return: w
        """
        print('load data ...')
        traindata, labels = self.load_data('train')
        # 初始化矩阵，这里矩阵初始化用的是均匀分布，也可以用全0，但是效果欠佳；
        w = torch.nn.init.uniform_(torch.Tensor(1, traindata.size(1)), a=0., b=1.)
        print('start train ...')
        for ep in tqdm(range(self.epoch)):
            for it in range(traindata.size(0)):
                w = self.logistic_regression(traindata[it], w, labels[it])
                print('epoch {} step {}'.format(ep, it))
        return w

    def test(self, w):
        """
        带标测试
        :param w: 权重，[1, hidden_size]
        :return: 正负样本准召率，以0.5为阈值
        """
        testdata, labels = self.load_data('test')
        res = []
        # 根据公式6.1.2（6.5），以0.5为阈值，大于等于的label=1，小于的label=0
        for it in range(testdata.size(0)):
            p = np.exp(torch.matmul(w, testdata[it])) / (1 + np.exp(torch.matmul(w, testdata[it])))
            if p >= 0.5:
                res.append(1)
            else:
                res.append(0)
        pn, nn = 0, 0
        ppred_d, npred_d = 0, 0
        prec_d, nrec_d = dict(Counter(labels))[1], dict(Counter(labels))[0]
        for i in range(len(res)):
            if res[i] == labels[i]:
                if res[i] == 1:
                    pn += 1
                else:
                    nn += 1
            if res[i] == 1:
                ppred_d += 1
            else:
                npred_d += 1
        print('正样本准确率/召回率: {}/{}\n'.format(pn / ppred_d if ppred_d else 0, pn / prec_d),
               '负样本准确率/召回率: {}/{}\n'.format(nn / npred_d if npred_d else 0, nn / nrec_d))

    def predict(self, w):
        """
        无标测试
        :param w:
        :return: 预测结果
        """
        testdata = self.load_data(opt='predict')
        res = []
        for it in range(testdata.size(0)):
            p = np.exp(torch.matmul(w, testdata[it])) / (1 + np.exp(torch.matmul(w, testdata[it])))
            if p >= 0.5:
                res.append(1)
            else:
                res.append(0)
        return res

if __name__ == '__main__':
    LR = LogisticRgression(fpath='')
    w = LR.train()
    LR = LogisticRgression(
        fpath='')
    LR.test(w)
