import LinearRegression.util.basic_node as nodes
import numpy as np


class affine:
    def __init__(self):
        self._feature_dim = 1
        self._Th = None

    def node_imp(self):
        self._node1 = nodes.mul_node()
        self._node2 = nodes.plus_node()

    def random_initialization(self):
        r_feature_dim = np.sqrt(1 / self._feature_dim)

        self._Th = np.random.uniform(
            low=-r_feature_dim,
            high=r_feature_dim,
            size=(self._feature_dim + 1, 1))

    def forward(self, x):
        self._z1 = self._node1.forward(self._Th[1], x)
        self._z2 = self._node2.forward(self._Th[0], self._z1)

        return self._z2

    def backward(self, dz, lr):
        dth0, dz1 = self._node2.backward(dz)
        dth1, _ = self._node1.backward(dz1)

        self._Th[1] -= lr * dth1
        self._Th[0] -= lr * dth0

    def get_Th(self):
        return self._Th


class Sigmoid:
    def __init__(self):
        self._pred = None

    def forward(self, z):
        self._pred = 1 / (1 + np.exp(-z))
        return self._pred

    def backward(self, dpred):
        partial = self._pred * (1-self._pred)
        dz = dpred * partial

        return dz


class BinaryCrossEntropy_Loss:
    def __init__(self):
        self._y, self._pred = None, None

    def forward(self, y, pred):
        self._y = y
        self._pred = pred

        loss = -1 * (y * np.log(self._pred) + (1 - y) * np.log(1 - pred))
        return loss

    def backward(self):
        dpred = (self._pred - self._y) / (self._pred * (1-self._pred))
        return dpred

