import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from SVLoR import *

import LinearRegression.util.basic_node as nodes
from dataset_generator import dataset_generator

np.random.seed(0)

plt.style.use('ggplot')
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 30
mpl.rcParams['axes.titlesize'] = 40
mpl.rcParams['legend.fontsize'] = 30

x_dict = {'mean': 1, 'std': 1, 'n_sample': 300, 'noise_factor': 0.3, 'cutoff': 1, 'direction': -1}
data = dataset_generator(x_dict)

affine = Affine()
sigmoid = Sigmoid()
BCE_loss = BinaryCrossEntropy_Loss()

th_accum = affine.get_Th()
loss_list = []

iter_idx, check_freq = 0, 2
epochs, lr = 300, 0.01

#learning
for epoch in range(epochs):
    np.random.shuffle(data)

    for data_idx in range(data.shape[0]):
        x, y = data[data_idx, 1], data[data_idx, -1],

        # forward propagation
        z = affine.forward(x)
        pred = sigmoid.forward(z)
        loss = BCE_loss.forward(y, pred)

        # back propagation
        dpred = BCE_loss.backward()
        dz = sigmoid.backward(dpred)
        affine.backward(dz, lr)

        if iter_idx % check_freq == 0:
            th_accum = np.hstack((th_accum, affine.get_Th()))
            loss_list.append(loss)
        iter_idx += 1

fig, ax = plt.subplots(2, 1, figsize=(30, 10))
fig.subplots_adjust(hspace=0.3)
ax[0].set_title(r'$\vec{\theta}$' + ' Update')
ax[0].plot(th_accum[1, :], label=r'$\theta_{1}$')
ax[0].plot(th_accum[0, :], label=r'$\theta_{0}$')
ax[0].legend()
iter_ticks = np.linspace(0, th_accum.shape[1], 10).astype(int)
ax[0].set_xticks(iter_ticks)
ax[1].set_title('Loss Decrease')
ax[1].plot(loss_list)
ax[1].set_xticks(iter_ticks)
n_pred = 1000
fig, ax = plt.subplots(figsize=(30, 10))
ax.set_title('Predictor Update')
ax.scatter(data[:, 1], data[:, -1])
ax_idx_arr = np.linspace(0, len(loss_list) - 1, n_pred).astype(int)
cmap = plt.get_cmap('rainbow', lut=len(ax_idx_arr))
x_pred = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 1000)
for ax_cnt, ax_idx in enumerate(ax_idx_arr):
    z = th_accum[1, ax_idx] * x_pred + th_accum[0, ax_idx]
    a = 1 / (1 + np.exp(-1 * z))
    ax.plot(x_pred, a,
            color=cmap(ax_cnt),
            alpha=0.2)
y_ticks = np.round(np.linspace(0, 1, 7), 2)
ax.set_yticks(y_ticks)
plt.show()