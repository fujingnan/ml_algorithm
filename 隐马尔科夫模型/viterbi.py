"""
主题: 手撕维特比算法
作者: 傅景楠
作者Blog：https://blog.csdn.net/qq_36583400
参考文献: 统计学习方法-II 李航
- 本代码利用隐马尔科夫模型有监督学习与维特比算法实现一个基础分词器；
- 在模型训练阶段，由于我们手中的数据既有观测值（分词结果）也有状态值（分词标志），所以直接采用课本【10.3.1】节提到的有监督方法估计模型参数即可
- 如果手里的数据只有观测而无状态，那么就只能采用Baum-Welch算法估计模型参数
大家可参照本人博客介绍阅读代码：https://blog.csdn.net/qq_36583400/article/details/128965203
"""
import numpy as np


class AttrDict(dict):
    # 一个小trick，将结果返回成一个字典格式
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def dataloader(datapath):
    with open(datapath, 'r') as reader:
        for line in reader:
            yield line


class Model:
    """
    模型的参数估计，非Baum Welch算法，而是采用有监督的统计方法
    """
    def __init__(self, trainfile, N, M, Q):
        """
        初始化一些参数
        :param trainfile: 训练集路径
        :param N: 所有可能的状态数
        :param M: 所有可能的观测数
        :param Q: 所有可能的状态
        """
        self.trainfile = trainfile
        self.N = N
        self.M = M
        self.Pi = np.zeros(N)
        self.A = np.zeros((N, N))
        self.B = np.zeros((N, M))
        # 用id来表示每个状态
        self.Q2id = {x: i for i, x in enumerate(Q)}

    def cal_rate(self):
        """
        通过【10.3.1】节的内容来计算π、A、B中各个元素的频数；
        :return:
        """
        reader = dataloader(self.trainfile)
        for i, line in enumerate(reader):
            line = line.strip().strip('\n')
            if not line:
                continue
            word_list = line.split(' ')
            status_sequence = []
            # 计算π和B中每个元素的频数
            for j, item in enumerate(word_list):
                if len(item) == 1:
                    flag = 'S'
                else:
                    flag = 'B' + 'M' * (len(item) - 2) + 'E'
                if j == 0:
                    # 初始状态π的值是每条样本第一个字的状态出现的次数；
                    self.Pi[self.Q2id[flag[0]]] += 1
                for t, s in enumerate(flag):
                    # B有几行就代表有几种状态，每一列代表该状态下每种观测生成的次数；
                    self.B[self.Q2id[s]][ord(item[t])] += 1
                # 构建状态序列
                status_sequence.extend(flag)
            # 计算A元素的频数
            for t, s in enumerate(status_sequence):
                # A[i][j]表示由上一时刻的状态i转移到当前时刻状态j的次数
                prev = status_sequence[t - 1]
                self.A[self.Q2id[prev]][self.Q2id[s]] += 1

    def generate_model(self):
        """
        构建模型参数：
        主要是将频数表示的模型参数转化成频率表示的模型参数，在本代码中，利用"频数/总数"来表示各个参数中的值，取log是为了将乘法计算改为加法计算，
        这样可以便于计算，且防止乘积过小的情况；
        :return:
        """
        self.cal_rate()
        norm = -2.718e+16
        denominator = sum(self.Pi)
        for i, pi in enumerate(self.Pi):
            if pi == 0.:
                self.Pi[i] = norm
            else:
                self.Pi[i] = np.log(pi / denominator)
        # 公式【10.30】
        for row in range(self.A.shape[0]):
            denominator = sum(self.A[row])
            for col, a in enumerate(self.A[row]):
                if a == 0.:
                    self.A[row][col] = norm
                else:
                    self.A[row][col] = np.log(a / denominator)
        # 公式【10.31】
        for row in range(self.B.shape[0]):
            denominator = sum(self.B[row])
            for col, b in enumerate(self.B[row]):
                if b == 0.:
                    self.B[row][col] = norm
                else:
                    self.B[row][col] = np.log(b / denominator)
        return AttrDict(
            pi=self.Pi,
            A=self.A,
            B=self.B
        )


class Viterbi:
    def __init__(self, model: dict):
        """
        初始化一些参数
        :param model: 由训练而成的模型作为维特比算法预测依据
        """
        self.pi = model.pi
        self.A = model.A
        self.B = model.B

    def predict(self, datapath):
        """
        根据算法【10.5】生成预测序列
        :param datapath: 测试集路径
        :return:
        """
        reader = dataloader(datapath)
        self.O = [line.strip().strip('\n') for line in reader]
        N = self.pi.shape[0]
        self.segs = []
        for o in self.O:
            o = [w for w in o if w]
            if not o:
                self.segs.append([])
                continue
            T = len(o)
            # 定义δ和ψ
            delta_t = np.zeros((T, N))
            psi_t = np.zeros((T, N))
            for t in range(T):
                if not t:
                    # t=1时，根据算法【10.5】第（1）步，计算δ_{1}和ψ_{1}
                    delta_t[t][:] = self.pi + self.B.T[:][ord(o[0])]  # 由于log转换，所以原先的*变成+
                    psi_t[t][:] = np.zeros((1, N))
                else:
                    # 根据算法【10.5】第（2）步，递推计算δ_{t}和ψ_{t}
                    deltaTemp = delta_t[t - 1] + self.A.T
                    for i in range(N):
                        delta_t[t][i] = max(deltaTemp[:][i]) + self.B[i][ord(o[t])]
                        psi_t[t][i] = np.argmax(deltaTemp[:][i])
            I = []
            # 当计算完所有δ和ψ后，找到T时刻的δ中的最大值的索引，即算法【10.5】第（3）步中的i*_{T}
            maxNode = np.argmax(delta_t[-1][:])
            I.append(int(maxNode))
            for t in range(T - 1, 0, -1):
                # 算法【10.5】第（4）步，回溯找i*_{t}
                maxNode = int(psi_t[t][maxNode])
                I.append(maxNode)
            I.reverse()
            self.segs.append(I)

    def segment(self):
        """
        根据状态序列对句子进行分词
        :return: 分词结果列表
        """
        segments = []
        for i, line in enumerate(self.segs):
            curText = ""
            temp = []
            for j, w in enumerate(line):
                if w == 0:
                    # 如果该字的状态为"S"，为单字
                    temp.append(self.O[i][j])
                else:
                    if w != 3:
                        # 如果该字的状态不为"E"，那么要么为"B"，要么为"M"，说明一个词还没结束；
                        curText += self.O[i][j]
                    else:
                        # 遇到结束状态符"E"时，该词分词结束；
                        curText += self.O[i][j]
                        temp.append(curText)
                        curText = ''
            segments.append(temp)
        return segments


if __name__ == '__main__':
    # 我们用编码表示汉字字符，用`ord()`方法获得汉字编码，所以构建所有可能观测值的数为65536，保证所有字都能覆盖到；
    # S：单字表示符；
    # B：一个词的起始符；
    # M：一个属于一个词中间字的标识；
    # E：一个词的结束符；
    trainer = Model(N=4, M=65536, Q=['S', 'B', 'M', 'E'], trainfile='train.txt')
    model = trainer.generate_model()
    segment = Viterbi(model)
    segment.predict('test.txt')
    print(segment.segment())
