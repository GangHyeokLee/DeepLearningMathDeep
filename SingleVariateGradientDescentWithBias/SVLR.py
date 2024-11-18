import basic_node as nodes
import numpy as np
from bias_plt_show import plt_show
from matplotlib import pyplot as plt

check_freq = 100


class SVLR:
    # th1, th0 초기화 여기서부터 시작
    def __init__(self, th1, th0):
        self.th1, self.th0 = th1, th0

        self.th1_list, self.th0_list = [], []
        self.cost_list = []

        self.iter_cnt, self.check_cnt = 0, 0

        self.model_imp()
        self.cost_imp()

    def model_imp(self):
        self.node1 = nodes.mul_node()
        self.node2 = nodes.plus_node()

    def cost_imp(self):
        self.node3 = nodes.minus_node()
        self.node4 = nodes.square_node()
        self.node5 = nodes.mean_node()

    def forward(self, mini_batch):
        Z1 = self.node1.forward(self.th1, mini_batch[:, 0])
        Z2 = self.node2.forward(Z1, self.th0)
        Z3 = self.node3.forward(mini_batch[:, 1], Z2)
        L = self.node4.forward(Z3)
        J = self.node5.forward(L)

        if self.iter_cnt % check_freq == 0:
            self.cost_list.append(J)

    def backward(self, lr):
        if self.iter_cnt % check_freq == 0:
            self.th1_list.append(self.th1)
            self.th0_list.append(self.th0)
            self.check_cnt += 1

        dL = self.node5.backward(1)
        dZ3 = self.node4.backward(dL)
        dY, dZ2 = self.node3.backward(dZ3)
        dZ1, dTh0 = self.node2.backward(dZ2)
        dTh1, dX = self.node1.backward(dZ1)

        self.th1 -= lr * np.sum(dTh1)
        self.th0 -= lr * np.sum(dTh0)

        self.iter_cnt += 1

    def result_visualization(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(2, 1, figsize=(30, 15))
        ax[0].plot(self.th1_list, label=r'$\theta_{1}$')
        ax[0].plot(self.th0_list, label=r'$\theta_{0}$')
        ax[1].plot(self.cost_list)
        ax[0].legend(loc='lower right', fontsize=30)

        fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.1)
        ax[0].axes.get_xaxis().set_visible(False)

        x_ticks = np.linspace(0, self.check_cnt, 10).astype(int)
        x_ticklabels = x_ticks * check_freq
        y_ticks = np.arange(0, self.th0 + 0.5)

        ax[1].set_xticks(x_ticks)
        ax[1].set_xticklabels(x_ticklabels)
        ax[0].set_yticks(y_ticks)

        ax[0].tick_params(axis='both', labelsize=40)
        ax[1].tick_params(axis='both', labelsize=40)

