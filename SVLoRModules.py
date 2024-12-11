import LinearRegression.util.basic_node as nodes
import numpy as np
import matplotlib.pyplot as plt


class Affine:
    def __init__(self):
        self._feature_dim = 1
        self._Th = None

        self.node_imp()
        self.random_initialization()

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
        partial = self._pred * (1 - self._pred)
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
        dpred = (self._pred - self._y) / (self._pred * (1 - self._pred))
        return dpred


def get_xdict(mean, std, n_sample, noise_factor, cutoff, direction):
    return {'mean': mean, 'std': std, 'n_sample': n_sample, 'noise_factor': noise_factor, 'cutoff': cutoff,
            'direction': direction}


def result_tracker(iter_idx, check_freq, th_accum, model, loss_list, loss):
    if iter_idx % check_freq == 0:
        th_accum = np.hstack((th_accum, model.get_Th()))
        loss_list.append(loss)
    iter_idx += 1
    return iter_idx, th_accum


def result_visualizer(th_accum, loss_list, data):
    fig, ax = plt.subplots(2, 1, figsize=(30, 10))
    fig.subplots_adjust(hspace=0.3)
    ax[0].set_title(r'$\vec{\theta}$' + ' Update')
    ax[0].plot(th_accum[1, :], label=r'$\theta_{1}$')
    ax[0].plot(th_accum[0, :], label=r'$\theta_{0}$')
    ax[0].legend()
    iter_ticks = np.linspace(0, th_accum.shape[1], 10).astype(np.int64)
    ax[0].set_xticks(iter_ticks)
    ax[1].set_title(r'$\mathcal{L}$')
    ax[1].plot(loss_list)
    ax[1].set_xticks(iter_ticks)
    n_pred = 1000
    fig, ax = plt.subplots(figsize=(30, 10))
    ax.set_title('Predictor Update')
    ax.scatter(data[:, 1], data[:, -1])
    ax_idx_arr = np.linspace(0, len(loss_list) - 1, n_pred).astype(np.int64)
    cmap = plt.get_cmap('rainbow', lut=len(ax_idx_arr))
    x_pred = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 1000)

    for ax_cnt, ax_idx in enumerate(ax_idx_arr):
        z = th_accum[1, ax_idx] * x_pred + th_accum[0, ax_idx]
        a = 1 / (1 + np.exp(-1 * z))
        ax.plot(x_pred, a,
                color=cmap(ax_cnt),
                alpha=
                0.2)
    y_ticks = np.round(np.linspace(0, 1, 7), 2)
    ax.set_yticks(y_ticks)

    plt.show()
