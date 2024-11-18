import numpy as np
import matplotlib.pyplot as plt

from dataset_generator import dataset_generator
import basic_node as nodes
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
epochs = 200
for _ in range(epochs):
    X, Y = x_data, y_data

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