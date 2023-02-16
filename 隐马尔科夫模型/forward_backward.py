import numpy as np

class FB:
    def __init__(self, pi, A, B, V):
        self.pi = pi
        self.A = A
        self.B = B
        self.V = V

    def forward(self, O):
        alpha_t_plus_1 = []
        for t, o in enumerate(O):
            if t == 0:
                alpha1 = []
                for i, p in enumerate(self.pi):
                    obj_index = self.V.index(o)
                    alpha1.append(p * self.B[i][obj_index])
                alpha_t_plus_1 = alpha1
            else:
                alpha_temp = []
                for i in range(self.A.shape[0]):
                    alpha_ji = 0.
                    for j, a in enumerate(alpha_t_plus_1):
                        alpha_ji += (a * self.A[j][i])
                    obj_index = self.V.index(o)
                    alpha_temp.append(alpha_ji * self.B[i][obj_index])
                alpha_t_plus_1 = alpha_temp
        P = sum(alpha_t_plus_1)
        return P

    def backward(self, O):
        beta1 = 1



if __name__ == '__main__':
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
    print(f.forward(['红', '白', '红']))