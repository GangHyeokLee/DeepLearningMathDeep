import numpy as np
from matplotlib import pyplot as plt

import basic_node as nodes


class Affine:
    def __init__(self):
        self._feature_dim = 1

        self.node_imp()
        self.random_initialization()

    def node_imp(self):
        self._node1 = nodes.mul_node()
        self._node2 = nodes.plus_node()

    def random_initialization(self):
        r_feature_dim = 1 / self._feature_dim
        self._Th = np.random.uniform(
            low=-1 * r_feature_dim,
            high=1 * r_feature_dim,
            size=(self._feature_dim + 1, 1)
        )

    def forward(self, X):
        self._Z1 = self._node1.forward(self._Th[1], X)
        self._Z2 = self._node2.forward(self._Th[0], self._Z1)
        return self._Z2

    def backward(self, dZ, lr):
        dTh0, dZ1 = self._node2.backward(dZ)
        dTh1, dX = self._node1.backward(dZ1)

        self._Th[1] -= lr * np.sum(dTh1)
        self._Th[0] -= lr * np.sum(dTh0)

    def get_Th(self):
        return self._Th


class Sigmoid:
    def __init__(self):
        self._Pred = None

    def forward(self, Z):
        self._Pred = 1 / (1 + np.exp(-Z))
        return self._Pred

    def backward(self, dPred):
        Partial = self._Pred * (1 - self._Pred)
        dZ = dPred * Partial
        return dZ


class SVLoR:
    def __init__(self):
        self._feature_dim = 1

        self._affine = Affine()
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
        Loss = -1 * (Y * np.log(self._Pred) + (1 - Y) * np.log(1 - Pred))
        J = self._mean_node.forward(Loss)
        return J

    def backward(self):
        dLoss = self._mean_node.backward(1)
        dPred = dLoss * (self._Pred - self._Y) / (self._Pred * (1 - self._Pred))
        return dPred


def result_tracker(iter_idx, check_freq, th_accum, model, cost_list, cost):
    if iter_idx % check_freq == 0:
        th_accum = np.hstack((th_accum, model.get_Th()))
        cost_list.append(cost)
    iter_idx += 1
    return iter_idx, th_accum


def result_visualizer(th_accum, cost_list, data, mean, std, n_sample, noise_factor, batch_size, epochs, lr):
    fig, ax = plt.subplots(3, 1, figsize=(15, 15))  # 세로로 3개의 그래프 배치
    fig.subplots_adjust(hspace=0.5)

    # 첫 번째 그래프: Theta Update
    ax[0].set_title(r'$\vec{\theta}$' + ' Update', fontsize=20)
    ax[0].plot(th_accum[1, :], label=r'$\theta_{1}$')
    ax[0].plot(th_accum[0, :], label=r'$\theta_{0}$')
    ax[0].legend()
    iter_ticks = np.linspace(0, th_accum.shape[1], 10).astype(int)
    ax[0].set_xticks(iter_ticks)

    # 두 번째 그래프: Cost
    ax[1].set_title('Cost')
    ax[1].plot(cost_list)
    ax[1].set_xticks(iter_ticks)

    # 세 번째 그래프: Predictor Update
    n_pred = 1000
    ax[2].set_title('Predictor Update')
    ax[2].scatter(data[:, 1], data[:, -1], label='Data')
    ax_idx_arr = np.linspace(0, len(cost_list) - 1, n_pred).astype(int)
    cmap = plt.get_cmap('rainbow', lut=len(ax_idx_arr))
    x_pred = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 1000)

    for ax_cnt, ax_idx in enumerate(ax_idx_arr):
        z = th_accum[1, ax_idx] * x_pred + th_accum[0, ax_idx]
        a = 1 / (1 + np.exp(-1 * z))
        ax[2].plot(x_pred, a, color=cmap(ax_cnt), alpha=0.2)

    y_ticks = np.round(np.linspace(0, 1, 7), 2)
    ax[2].set_yticks(y_ticks)

    # 세 번째 그래프에 파라미터 정보 표시
    param_text = (f"mean: {mean}\nstd: {std}\nn_sample: {n_sample}\n"
                  f"noise_factor: {noise_factor}\nbatch_size: {batch_size}\n"
                  f"epochs: {epochs}\nlr: {lr}")
    ax[2].text(0.02, 0.95, param_text, transform=ax[2].transAxes,
               fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.show()


def plot_classifier_with_projection(data, th_accum):
    # 데이터 분리
    p_idx = np.where(data[:, -1] > 0)
    np_idx = np.where(data[:, -1] <= 0)
    
    # 결정 경계면 계산
    f_th0, f_th1 = th_accum[:, -1]  # th0, th1만 사용
    x_range = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
    
    sigmoid = Sigmoid()
    # 시그모이드 함수 계산
    z = f_th1 * x_range + f_th0
    pred = sigmoid.forward(z)
    
    # 그래프 생성
    fig = plt.figure(figsize=(20, 8))
    
    # 3D 그래프
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(data[p_idx, 1].flat, np.zeros_like(data[p_idx, 1].flat), data[p_idx, -1].flat, 
               c='blue', marker='o', label='Positive Class')
    ax1.scatter(data[np_idx, 1].flat, np.zeros_like(data[np_idx, 1].flat), data[np_idx, -1].flat, 
               c='red', marker='x', label='Negative Class')
    
    # 결정 경계면 그리기
    X = x_range
    Y = np.zeros_like(x_range)  # y축은 0으로 고정
    Z = pred
    ax1.plot(X, Y, Z, color='g', linewidth=2, label='Decision Boundary')
    
    ax1.set_xlabel(r'$x$', labelpad=10)
    ax1.set_ylabel('', labelpad=10)  # y축 레이블 제거
    ax1.set_zlabel('Probability', labelpad=10)
    ax1.set_title('3D Logistic Regression Result')
    ax1.legend()
    
    # 2D 그래프
    ax2 = fig.add_subplot(122)
    ax2.scatter(data[p_idx, 1].flat, data[p_idx, -1].flat, c='blue', marker='o', label='Positive Class')
    ax2.scatter(data[np_idx, 1].flat, data[np_idx, -1].flat, c='red', marker='x', label='Negative Class')
    ax2.plot(x_range, pred, color='g', linewidth=2, label='Logistic Function')
    
    # 결정 경계선 (y=0.5)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision Boundary (y=0.5)')
    
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel('Probability')
    ax2.set_title('Logistic Regression Decision Boundary')
    ax2.legend()
    
    # 파라미터 정보 표시
    param_text = f"θ₀: {f_th0:.3f}\nθ₁: {f_th1:.3f}"
    ax2.text(0.02, 0.95, param_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()