import numpy as np
import matplotlib.pyplot as plt

from LinearRegression.util.dataset_generator import dataset_generator
from LinearRegression.util import basic_node as nodes
from LinearRegression.SingleVariateGradientDescentWithoutBias.plt_show import plt_show

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

th = -1
lr = 0.01

loss_list, th_list = [], []
iterations = 200
for _ in range(iterations):
    idx_arr = np.arange(len(x_data))
    random_idx = np.random.choice(idx_arr, 1)

    x, y = x_data[random_idx], y_data[random_idx]

    z1 = node1.forward(th, x)
    z2 = node2.forward(y, z1)
    L = node3.forward(z2)

    dZ2 = node3.backward(1)
    dy, dZ1 = node2.backward(dZ2)
    dth, dx = node1.backward(dZ1)

    th -= lr * dth
    loss_list.append(L.item())
    th_list.append(th.item())


plt_show(th_list, loss_list, x_data, y_data)