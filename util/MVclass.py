from util import basic_node as nodes
import numpy as np
from matplotlib import pyplot as plt


class Affine_Function:
    def __init__(self, feature_dim, Th):
        self._node2 = None
        self._node1 = None
        self._feature_dim = feature_dim
        self._Th = Th

        self._z1_list = [None] * (self._feature_dim + 1)
        self._z2_list = [None] * (self._feature_dim + 1)

        self._dz2_list = [None] * (self._feature_dim + 1)
        self._dz1_list = [None] * (self._feature_dim + 1)

        self._dth_list = [None] * (self._feature_dim + 1)

    def affine_imp(self):
        self._node1 = [None] + [nodes.mul_node() for _ in range(self._feature_dim)]
        self._node2 = [None] + [nodes.plus_node() for _ in range(self._feature_dim)]

    def forward(self, X):
        for node_idx in range(1, self._feature_dim + 1):
            self._z1_list[node_idx] = self._node1[node_idx].forward(self._Th[node_idx], X[node_idx])

        self._z2_list[1] = self._node2[1].forward(self._Th[0], self._z1_list[1])

        for node_idx in range(2, self._feature_dim + 1):
            self._z2_list[node_idx] = self._node2[node_idx].forward(self._z2_list[node_idx - 1],
                                                                    self._z1_list[node_idx])
        return self._z2_list[-1]

    def backward(self, dz2_last, lr):
        self._dz2_list[-1] = dz2_last

        for node_idx in reversed(range(1, self._feature_dim + 1)):
            self._dz2_list[node_idx - 1], self._dz1_list[node_idx] = self._node2[node_idx].backward(
                self._dz2_list[node_idx])

        self._dth_list[0] = self._dz2_list[0]

        for node_idx in reversed(range(1, self._feature_dim + 1)):
            dth, _ = self._node1[node_idx].backward(self._dz1_list[node_idx])
            self._dth_list[node_idx] = dth

        self._Th -= lr * np.array(self._dth_list).reshape(-1, 1)
        return self._Th


class Square_Error_Loss:
    def __init__(self):
        self.loss_imp()

    def loss_imp(self):
        self._node3 = nodes.minus_node()
        self._node4 = nodes.square_node()

    def forward(self, y, pred):
        z3 = self._node3.forward(y, pred)
        l = self._node4.forward(z3)
        return l

    def backward(self):
        dz3 = self._node4.backward(1)
        _, dz2_last = self._node3.backward(dz3)

        return dz2_last


def result_visualization(th_accum, loss_list, feature_dim):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(2, 1, figsize=(40, 20))

    for i in range(feature_dim + 1):
        ax[0].plot(th_accum[i], label=r'$\theta_{%d}$' % i, linewidth=5)

    ax[1].plot(loss_list)

    ax[0].legend(loc='lower right', fontsize=30)
    ax[0].tick_params(axis='both', labelsize=30)
    ax[1].tick_params(axis='both', labelsize=30)

    ax[0].set_title(r'$\vec{\theta}$', fontsize=40)
    ax[1].set_title('Loss', fontsize=40)

    plt.show()
