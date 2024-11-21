# import required modules
import matplotlib.pyplot as plt
import numpy as np

from util.dataset_generator import dataset_generator
from util import basic_node as nodes

# dataset preparation
dataset_gen = dataset_generator()
dataset_gen.set_coefficient([5, 0])

x_data, y_data = dataset_gen.make_dataset()
print(x_data.shape)
print(y_data.shape)
dataset_gen.dataset_visualizer(x_data, y_data)

# model part
node1 = nodes.mul_node()

#square error and MSE cost part
node2 = nodes.minus_node()
node3 = nodes.square_node()
node4 = nodes.mean_node()

# hyperparameter setting
epochs = 50
lr = 0.01

th = -1
cost_list = []
th_list = []

for epoch in range(epochs):
    X, Y = x_data, y_data
    Z1 = node1.forward(th, X)
    Z2 = node2.forward(Y, Z1)
    L = node3.forward(Z2)
    J = node4.forward(L)

    dL = node4.backward(J)
    dZ2 = node3.backward(dL)
    dY, dZ1 = node2.backward(dZ2)
    dTh, dX = node1.backward(dZ1)

    th = th - lr * np.sum(dTh)

    th_list.append(th)
    cost_list.append(J)

fig, ax = plt.subplots(2, 1, figsize=(30, 10))
ax[0].plot(th_list)
ax[1].plot(cost_list)

title_font = {'size': 30, 'alpha': 0.8, 'color': 'navy'}
label_font = {'size': 20, 'alpha': 0.8}
plt.style.use('seaborn-v0_8-darkgrid')

ax[0].set_title(r'$\theta$', fontdict=title_font)
ax[1].set_title("Cost", fontdict=title_font)
ax[1].set_xlabel("Epoch", fontdict=label_font)

plt.show()

N_line = min(100, len(th_list))
cmap = plt.get_cmap('rainbow', lut=N_line)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(x_data, y_data)

test_th = th_list[:N_line]
x_range = np.array([np.min(x_data), np.max(x_data)])

for line_idx in range(N_line):
    pred_line = np.array([x_range[0] * test_th[line_idx], x_range[1] * test_th[line_idx]])
    ax.plot(x_range, pred_line, color=cmap(line_idx), alpha=0.1)

plt.show()