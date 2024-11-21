import numpy as np
import matplotlib.pyplot as plt

from util.dataset_generator import dataset_generator
from util import basic_node as nodes
from SingleVariateGradientDescentWithoutBias.plt_show import plt_show

np.random.seed(0)
plt.style.use('ggplot')

# Dataset Preparation
dataset_gen = dataset_generator()
dataset_gen.set_coefficient([5, 0])
x_data, y_data = dataset_gen.make_dataset()
dataset_gen.dataset_visualizer(x_data, y_data)

# model part
node1 = nodes.mul_node()

# Square Error Loss Part
node2 = nodes.minus_node()
node3 = nodes.square_node()
node4 = nodes.mean_node()

th = -1
lr = 0.01
loss_list, th_list = [], []

batch_size = 16
n_batch = int(np.ceil(len(x_data) / batch_size))
t_iteration = 500
epochs = np.ceil(t_iteration / n_batch).astype(int)

for epoch in range(epochs):

    idx_np = np.arange(len(x_data))
    np.random.shuffle(idx_np)
    x_data = x_data[idx_np]
    y_data = y_data[idx_np]

    for batch_idx in range(n_batch):
        if batch_idx is n_batch - 1:
            X = x_data[batch_idx * batch_size:]
            Y = y_data[batch_idx * batch_size:]
        else:
            X = x_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            Y = y_data[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        z1 = node1.forward(th, X)
        z2 = node2.forward(Y, z1)
        L = node3.forward(z2)
        J = node4.forward(L)

        dL = node4.backward(1)
        dZ2 = node3.backward(dL)
        dy, dZ1 = node2.backward(dZ2)
        dth, dx = node1.backward(dZ1)

        th -= lr * np.sum(dth)
        loss_list.append(J)
        th_list.append(th)

plt_show(th_list, loss_list, x_data, y_data)
