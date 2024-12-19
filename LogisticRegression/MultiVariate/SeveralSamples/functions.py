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

        self._dTh_list = [None] * (self._feature_dim + 1)

        self._node_imp()
        self._random_initialization()

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
        self._Pred = None

    def forward(self, Z):
        self._Pred = 1 / (1 + np.exp(-1 * Z))
        return self._Pred

    def backward(self, dPred):
        Partial = self._Pred * (1 - self._Pred)
        dZ = dPred * Partial
        return dZ


class Tanh:
    def __init__(self):
        self._Pred = None

    def forward(self, Z):
        self._Pred = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        return self._Pred

    def backward(self, dPred):
        Partial = (1 + self._Pred) * (1 - self._Pred)
        dZ = dPred * Partial
        return dZ


class ReLU:
    def __init__(self):
        self._Pred = None

    def forward(self, Z):
        self._Pred = Z
        return np.maximum(0, Z)

    def backward(self, dPred):
        dZ = dPred * (self._Pred > 0)
        return dZ


class AN:
    def __init__(self, feature_dim, activation='Sigmoid'):
        self._activation = activation
        self._activation_function = None

        self._feature_dim = feature_dim
        self._affine = Affine_Function(self._feature_dim)

        self._act_imp()

    def _act_imp(self):
        if self._activation == 'Sigmoid':
            self._activation_function = Sigmoid()
        elif self._activation == 'Tanh':
            self._activation_function = Tanh()
        elif self._activation == 'ReLU':
            self._activation_function = ReLU()
        else:
            print('unknown activation function')

    def forward(self, X):
        Z = self._affine.forward(X)
        Pred = self._activation_function.forward(Z)
        return Pred

    def backward(self, dPred, lr):
        dZ = self._activation_function.backward(dPred)
        self._affine.backward(dZ, lr)

    def get_Th(self):
        return self._affine.get_Th()


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


class BinaryCrossEntropy_Cost:
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


def plot_classifier_with_projection(data, th_accum, sigmoid):
    # 데이터 분리
    p_idx = np.where(data[:, -1] > 0)
    np_idx = np.where(data[:, -1] <= 0)
    
    # 결정 경계면 계산
    f_th0, f_th1, f_th2 = th_accum[:, -1]
    x1_range = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
    x2_range = np.linspace(np.min(data[:, 2]), np.max(data[:, 2]), 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # 예측 확률 계산
    Z = X2 * f_th2 + X1 * f_th1 + f_th0
    pred = sigmoid.forward(Z)
    
    # 그래프 생성
    fig = plt.figure(figsize=(20, 8))
    
    # 3D 그래프 (데이터 분포와 결정 경계면)
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 데이터 포인트 표시
    ax1.scatter(data[p_idx, 1].flat, data[p_idx, 2].flat, data[p_idx, -1].flat, 
                c='blue', marker='o', label='Positive Class')
    ax1.scatter(data[np_idx, 1].flat, data[np_idx, 2].flat, data[np_idx, -1].flat, 
                c='red', marker='x', label='Negative Class')
    
    # 결정 경계면 표시
    surf = ax1.plot_surface(X1, X2, pred, alpha=0.3, cmap='viridis')
    
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_zlabel('Probability')
    ax1.set_title('3D Data Distribution and Decision Surface')
    ax1.legend()
    
    # 컬러바 추가
    fig.colorbar(surf, ax=ax1, label='Prediction Probability')
    
    # 2D 그래프 (x₁-x₂ 평면에서의 결정 경계)
    ax2 = fig.add_subplot(122)
    
    # 데이터 포인트 표시
    ax2.scatter(data[p_idx, 1].flat, data[p_idx, 2].flat, 
                c='blue', marker='o', label='Positive Class')
    ax2.scatter(data[np_idx, 1].flat, data[np_idx, 2].flat, 
                c='red', marker='x', label='Negative Class')
    
    # 결정 경계 컨투어 표시
    contour = ax2.contour(X1, X2, pred, levels=[0.5], 
                         colors='g', linewidths=2)
    ax2.clabel(contour, inline=True, fmt='Decision Boundary')
    
    # 확률 분포 표시
    contourf = ax2.contourf(X1, X2, pred, levels=np.linspace(0, 1, 11), 
                           alpha=0.3, cmap='viridis')
    plt.colorbar(contourf, ax=ax2, label='Prediction Probability')
    
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_title('Decision Boundary in x₁-x₂ Plane')
    ax2.legend()
    
    # 파라미터 정보 표시
    param_text = f"θ₀: {f_th0:.3f}\nθ₁: {f_th1:.3f}\nθ₂: {f_th2:.3f}"
    ax2.text(0.02, 0.95, param_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()