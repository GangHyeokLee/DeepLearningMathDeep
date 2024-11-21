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

        self.affine_imp()

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

        for th_idx in range(self._Th.shape[0]):
            self._Th[th_idx] -= lr * np.sum(self._dth_list[th_idx])

        return self._Th

class MSE_Cost:
    def __init__(self):
        self.loss_imp()

    def loss_imp(self):
        self._node3 = nodes.minus_node()
        self._node4 = nodes.square_node()
        self._node5 = nodes.mean_node()

    def forward(self, y, pred):
        z3 = self._node3.forward(y, pred)
        z4 = self._node4.forward(z3)
        J = self._node5.forward(z4)
        return J

    def backward(self):
        dz4 = self._node5.backward(1)
        dz3 = self._node4.backward(dz4)
        _, dz2_last = self._node3.backward(dz3)

        return dz2_last
