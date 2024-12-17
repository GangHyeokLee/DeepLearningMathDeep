import numpy as np
from matplotlib import pyplot as plt

from LinearRegression.util import basic_node as nodes


class Affine_Function:
    def __init__(self, feature_dim):
        self._feature_dim = feature_dim

        self._Z1_list = [None] * (self._feature_dim + 1)
        self._Z2_list = [None] * (self._feature_dim + 1)

        self._dZ1_list = [None] * (self._feature_dim + 1)
        self._dZ2_list = [None] * (self._feature_dim + 1)

    def _node_imp(self):
        self._node1 = [None] + [nodes.mul_node() for _ in range(self._feature_dim)]
        self._node2 = [None] + [nodes.plus_node() for _ in range(self._feature_dim)]

    def _random_initialization(self):
        r_feature_dim = 1 / self._feature_dim
        self._Th = np.random.uniform(low=-r_feature_dim, high=r_feature_dim, size=(self._feature_dim + 1, 1))

    def forward(self, X):
        for node_idx in range(1, self._feature_dim + 1):
            self._Z1_list[node_idx] = self._node1[node_idx].forward(self._Th[node_idx], X[:, node_idx])

        self._Z2_list[1] = self._node2[1].forward(self._Th[0], self._Z1_list[1])
        for node_idx in range(2, self._feature_dim + 1):
            self._Z2_list[node_idx] = self._node2[node_idx].forward(self._Z2_list[node_idx - 1],
                                                                    self._Z1_list[node_idx])
        return self._Z2_list[-1]

    def backward(self, dZ2_last, lr):
        self._dZ2_list[-1] = dZ2_last

        for node_idx in reversed(range(1, self._feature_dim + 1)):
            dZ2, dZ1 = self._node2[node_idx].backward(self._dZ2_list[node_idx])
            self._dZ2_list[node_idx - 1] = dZ2
            self._dZ1_list[node_idx] = dZ1
        self._dTh_list[0] = self._dZ2_list[0]

        for node_idx in reversed(range(1, self._feature_dim + 1)):
            dTh, _ = self._node1[node_idx].backward(self._dZ1_list[node_idx])
            self._dTh_list[node_idx] = dTh

        for th_idx in range(self._Th.shape[0]):
            self._Th[th_idx] -= lr * np.sum(self._dTh_list[th_idx])

    def get_Th(self):
        return self._Th


class Sigmoid:
    def __init__(self):
        self._pred = None

    def forward(self, Z):
        self._Pred = 1 / (1 + np.exp(-1 * Z))
        return self._Pred

    def backward(self, dPred):
        Partial = self._Pred * (1 - self._Pred)
        dZ = dPred * Partial
        return dZ


class MVLoR:
    def __init__(self, feature_dim):
        self._feature_dim = feature_dim
        self._affine = Affine_Function(self._feature_dim)
        self._sigmoid = Sigmoid()

    def forward(self, X):
        Z = self._affine.forward(X)
        Pred = self._sigmoid.forward(Z)
        return Pred

    def backward(self, dPred, lr):
        dZ = self._sigmoid.backward(dPred)
        self._affine.backward(dZ, lr)

    def get_Th(self):
        return self._affine.get_Th()


class BinaryCrossEntropy_Loss:
    def __init__(self):
        self._Y, self._Pred = None, None
        self._mean_node = nodes.mean_node()

    def forward(self, Y, Pred):
        self._Y, self._Pred = Y, Pred
        Loss = -1 * (Y * np.log(self._Pred) + (1 - Y) * np.log(1 - self._Pred))
        J = self._mean_node.forward(Loss)
        return J

    def backward(self):
        dLoss = self._mean_node.backward(1)
        dPred = dLoss * (self._Pred - self._Y) / (self._Pred * (1 - self._Pred))
        return dPred


def plot_classifier(data, th_accum, sigmoid, ax):
    # Separate positive and negative classes
    p_idx = np.where(data[:, -1] > 0)
    np_idx = np.where(data[:, -1] <= 0)

    # Plot the positive and negative class points
    ax.plot(data[p_idx, 1].flat, data[p_idx, 2].flat, data[p_idx, -1].flat, 'bo', label='Positive Class')
    ax.plot(data[np_idx, 1].flat, data[np_idx, 2].flat, data[np_idx, -1].flat, 'rx', label='Negative Class')

    ax.set_xlabel(r'$x_{1}$' + ' data', labelpad=20)
    ax.set_ylabel(r'$x_{2}$' + ' data', labelpad=20)
    ax.set_zlabel('y', labelpad=20)

    # Decision boundary calculation
    f_th0, f_th1, f_th2 = th_accum[:, -1]
    x1_range = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
    x2_range = np.linspace(np.min(data[:, 2]), np.max(data[:, 2]), 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    affine = X2 * f_th2 + X1 * f_th1 + f_th0
    pred = sigmoid.forward(affine)

    # Plot decision boundary
    ax.plot_wireframe(X1, X2, pred, color='g', linewidth=2)

    ax.legend()


def get_data_batch(data, batch_idx, n_batch, batch_size):
    if batch_idx is n_batch - 1:
        batch = data[batch_idx * batch_size:]
    else:
        batch = data[batch_idx * batch_size: (batch_idx + 1) * batch_size]
    return batch


def result_tracker(iter_idx, check_freq, th_accum, affine, loss_list, loss):
    if iter_idx % check_freq == 0:
        th_accum = np.hstack((th_accum, affine.get_Th()))
        loss_list.append(loss)
    iter_idx += 1
    return iter_idx, th_accum


def result_visualizer(th_accum, feature_dim):
    fig, ax = plt.subplots(figsize=(30, 10))
    fig.subplots_adjust(hspace=0.3)
    ax.set_title(r'$\vec{\theta}$' + ' Update')

    for feature_idx in range(feature_dim + 1):
        ax.plot(th_accum[feature_idx, :], label=r'$\theta_{%d}$' % feature_idx)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    iter_ticks = np.linspace(0, th_accum.shape[1], 10).astype(int)
    ax.set_xticks(iter_ticks)

    plt.show()
