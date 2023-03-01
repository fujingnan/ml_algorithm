from baum_welch import Baum_Welch
import numpy as np
from LAC import LAC

lac = LAC(mode='lac')

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
    def __init__(self, trainfile, N, M, Q):
        self.trainfile = trainfile
        self.N = N
        self.M = M
        self.Pi = np.zeros(N)
        self.A = np.zeros((N, N))
        self.B = np.zeros((N, M))
        self.Q2id = {x: i for i, x in enumerate(Q)}

    def cal_rate(self):
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
                    self.Pi[self.Q2id[flag[0]]] += 1
                for t, s in enumerate(flag):
                    self.B[self.Q2id[s]][ord(item[t])] += 1
                status_sequence.extend(flag)
            # 计算A元素的频数
            for t, s in enumerate(status_sequence):
                prev = status_sequence[t - 1]
                self.A[self.Q2id[prev]][self.Q2id[s]] += 1

    def generate_model(self):
        self.cal_rate()
        norm = -2.718e+16
        denominator = sum(self.Pi)
        for i, pi in enumerate(self.Pi):
            if pi == 0.:
                self.Pi[i] = norm
            else:
                self.Pi[i] = np.log(pi / denominator)

        for row in range(self.A.shape[0]):
            denominator = sum(self.A[row])
            for col, a in enumerate(self.A[row]):
                if a == 0.:
                    self.A[row][col] = norm
                else:
                    self.A[row][col] = np.log(a / denominator)

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
    def __init__(self, model:dict):
        self.pi = model.pi
        self.A = model.A
        self.B = model.B

    def predict(self, datapath):
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
            delta_t = np.zeros((T, N))
            psi_t = np.zeros((T, N))
            for t in range(T):
                if not t:
                    delta_t[t][:] = self.pi + self.B.T[:][ord(o[0])] # 由于log转换，所以原先的*变成+
                    psi_t[t][:] = np.zeros((1, N))
                else:
                    deltaTemp = delta_t[t - 1] + self.A.T
                    for i in range(N):
                        delta_t[t][i] = max(deltaTemp[:][i]) + self.B[i][ord(o[t])]
                        psi_t[t][i] = np.argmax(deltaTemp[:][i])
            I = []
            maxNode = np.argmax(delta_t[-1][:])
            I.append(int(maxNode))
            for t in range(T - 1, 0, -1):
                maxNode = int(psi_t[t][maxNode])
                I.append(maxNode)
            I.reverse()
            self.segs.append(I)

    def segment(self):
        segments = []
        for i, line in enumerate(self.segs):
            curText = ""
            temp = []
            for j, w in enumerate(line):
                if w == 0:
                    temp.append(self.O[i][j])
                else:
                    if w != 3:
                        curText += self.O[i][j]
                    else:
                        curText += self.O[i][j]
                        temp.append(curText)
                        curText = ''
            segments.append(temp)
        return segments


if __name__ == '__main__':
    trainer = Model(N=4, M=65536, Q=['S', 'B', 'M', 'E'], trainfile='train.txt')
    model = trainer.generate_model()
    segment = Viterbi(model)
    segment.predict('test.txt')
    print(segment.segment())
