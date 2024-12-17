from matplotlib import pyplot as plt

from LinearRegression.util import basic_node as nodes
import numpy as np


class Affine_Function:
    def __init__(self, feature_dim):
        self._Th = None
        self._node2 = None
        self._node1 = None
        self._feature_dim = feature_dim

        self._z1_list = [None] * (self._feature_dim + 1)
        self._z2_list = [None] * (self._feature_dim + 1)

        self._dz1_list = [None] * (self._feature_dim + 1)
        self._dz2_list = [None] * (self._feature_dim + 1)

        self._dth_list = [None] * (self._feature_dim + 1)

        self.node_imp()
        self.random_initialization()

    def node_imp(self):
        self._node1 = [None] + [nodes.mul_node() for _ in range(self._feature_dim)]
        self._node2 = [None] + [nodes.plus_node() for _ in range(self._feature_dim)]

    def random_initialization(self):
        r_feature_dim = 1 / np.power(self._feature_dim, 0.5)
        self._Th = np.random.uniform(low=-1 * r_feature_dim, high=r_feature_dim, size=(self._feature_dim + 1, 1))

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
            dz2, dz1 = self._node2[node_idx].backward(self._dz2_list[node_idx])
            self._dz2_list[node_idx - 1] = dz2
            self._dz1_list[node_idx] = dz1

        self._dth_list[0] = self._dz2_list[0]

        for node_idx in reversed(range(1, self._feature_dim + 1)):
            dth, _ = self._node1[node_idx].backward(self._dz1_list[node_idx])
            self._dth_list[node_idx] = dth

        self._Th -= lr * np.array(self._dth_list).reshape(-1, 1)
        return self._Th

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
        self._y, self._pred = y, pred
        loss = -1 * (y * np.log(self._pred) + (1 - y) * np.log(1 - self._pred))
        return loss

    def backward(self):
        dpred = (self._pred - self._y) / (self._pred * (1 - self._pred))
        return dpred


def result_tracker(iter_idx, check_freq, th_accum, affine, loss_list, loss):
    if iter_idx % check_freq == 0:
        th_accum = np.hstack((th_accum, affine.get_Th()))
        loss_list.append(loss)
    iter_idx += 1
    return iter_idx, th_accum


# def plot_classifier(data, th_accum, sigmoid):
#     p_idx = np.where(data[:, -1] > 0)
#     np_idx = np.where(data[:, -1] <= 0)
#
#     fig = plt.figure(figsize=(15, 15))
#     ax = fig.add_subplot(projection='3d')
#
#     ax.plot(data[p_idx, 1].flat, data[p_idx, 2].flat, data[p_idx, -1].flat, 'bo')
#     ax.plot(data[np_idx, 1].flat, data[np_idx, 2].flat, data[np_idx, -1].flat, 'rx')
#
#     ax.set_xlabel(r'$x_{1}$' + ' data', labelpad=20)
#     ax.set_ylabel(r'$x_{2}$' + ' data', labelpad=20)
#     ax.set_zlabel('y', labelpad=20)
#
#     f_th0, f_th1, f_th2 = th_accum[:, -1]
#     x1_range = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
#     x2_range = np.linspace(np.min(data[:, 2]), np.max(data[:, 2]), 100)
#     X1, X2 = np.meshgrid(x1_range, x2_range)
#
#     affine = X2 * f_th2 + X1 * f_th1 + f_th0
#     pred = sigmoid.forward(affine)
#
#     ax.plot_wireframe(X1, X2, pred)
#
# def result_visualizer(th_accum, loss_list, feature_dim):
#     plt.style.use("ggplot")
#     fig, ax = plt.subplots(figsize=(30, 10))
#     fig.subplots_adjust(hspace=0.3)
#     ax.set_title(r'$\vec{\theta}$' + ' Update')
#
#     for feature_idx in range(feature_dim+1):
#         ax.plot(th_accum[feature_idx, :], label = r'$\theta_{%d}$'%feature_idx)
#
#     ax.legend()
#     iter_ticks = np.linspace(0, th_accum.shape[1], 10).astype(int)
#     ax.set_xticks(iter_ticks)
#
# def dataset_visualizer(data):
#     plt.style.use('ggplot')
#
#     # 데이터에서 클래스별로 인덱스를 찾기
#     p_idx = np.where(data[:, -1] > 0)  # 양성 클래스
#     np_idx = np.where(data[:, -1] <= 0)  # 음성 클래스
#
#     # 그래프 설정
#     fig = plt.figure(figsize=(15, 15))
#     ax = fig.add_subplot(projection='3d')
#
#     # 양성 클래스 (blue circle)과 음성 클래스 (red X) 점으로 그래프 그리기
#     ax.plot(data[p_idx, 1].flat, data[p_idx, 2].flat, data[p_idx, -1].flat, 'bo', label='Positive Class')
#     ax.plot(data[np_idx, 1].flat, data[np_idx, 2].flat, data[np_idx, -1].flat, 'rX', label='Negative Class')
#
#     # 축 레이블 추가
#     ax.set_xlabel(r'$x_{1}$' + ' data', labelpad=20)
#     ax.set_ylabel(r'$x_{2}$' + ' data', labelpad=20)
#     ax.set_zlabel('y', labelpad=20)
#
#     # 범례 추가
#     ax.legend()
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


def result_visualizer(th_accum, feature_dim, ax):
    ax.set_title(r'$\vec{\theta}$' + ' Update')
    for feature_idx in range(feature_dim + 1):
        ax.plot(th_accum[feature_idx, :], label=r'$\theta_{%d}$' % feature_idx)
    ax.legend()

    iter_ticks = np.linspace(0, th_accum.shape[1], 10).astype(int)
    ax.set_xticks(iter_ticks)


def dataset_visualizer(data, ax):
    # Separate positive and negative classes
    p_idx = np.where(data[:, -1] > 0)
    np_idx = np.where(data[:, -1] <= 0)

    # Plot the dataset with positive and negative points
    ax.plot(data[p_idx, 1].flat, data[p_idx, 2].flat, data[p_idx, -1].flat, 'bo', label='Positive Class')
    ax.plot(data[np_idx, 1].flat, data[np_idx, 2].flat, data[np_idx, -1].flat, 'rX', label='Negative Class')

    ax.set_xlabel(r'$x_{1}$' + ' data', labelpad=20)
    ax.set_ylabel(r'$x_{2}$' + ' data', labelpad=20)
    ax.set_zlabel('y', labelpad=20)

    ax.legend()


def visualize_all(data, th_accum, sigmoid, feature_dim):
    plt.style.use("ggplot")
    # Create a figure and axes for subplots
    fig = plt.figure(figsize=(15, 15))  # Large figure to accommodate multiple plots

    # # Subplot 1: Dataset Visualization
    # ax1 = fig.add_subplot(311, projection='3d')  # 1st plot in 3-row grid
    # dataset_visualizer(data, ax1)

    # Subplot 2: Theta Update Visualization
    ax2 = fig.add_subplot(211)  # 2nd plot in 3-row grid
    result_visualizer(th_accum, feature_dim, ax2)

    # Subplot 3: Classifier and Decision Boundary Visualization
    ax3 = fig.add_subplot(212, projection='3d')  # 3rd plot in 3-row grid
    plot_classifier(data, th_accum, sigmoid, ax3)

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()
