from baum_welch import Baum_Welch
import numpy as np
from LAC import LAC

lac = LAC(mode='lac')


def dataloader(datapath):
    with open(datapath, 'r') as reader:
        return reader


class Model(Baum_Welch):
    def __init__(self, N, M, V):
        super().__init__(N, M, V)

    def train_parameters(self, O, n):
        lam = self.train(O, n)
        return lam


class Viterbi:
    def __init__(self, pi, A, B):
        self.pi = pi
        self.A = A
        self.B = B

    def predict(self, O):
        self.O = O
        N = self.pi.shape[0]
        self.segs = []
        for o in O:
            T = len(o)
            delta_t = np.zeros((T, N))
            psi_t = np.zeros((T, N))
            for t in range(T):
                if not t:
                    delta_t[t][:] = self.pi * self.B[:][ord(o[0])]
                    psi_t[t][:] = np.zeros((1, N))
                else:
                    deltaTemp = delta_t[t - 1] * self.A.T
                    for i in range(N):
                        delta_t[t][i] = max(deltaTemp[:][i]) * self.B[i][ord(o[t])]
                        psi_t[t][i] = np.argmax(deltaTemp[:][i])
            I = []
            maxNode = np.argmax(delta_t[-1][:])
            I.append(maxNode)
            for t in range(T - 1, 0, -1):
                maxNode = psi_t[t][maxNode]
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
            segments.append(temp)
        return segments

if __name__ == '__main__':
    reader = dataloader('train.txt')
    V = [0] * 65536
    for line in reader:
        for w in line:
