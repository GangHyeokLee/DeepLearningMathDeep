import numpy as np

from util import basic_node as nodes
from util.dataset_generator import dataset_generator
from bias_plt_show import plt_show

np.random.seed(0)

dataset_gen = dataset_generator()
# y=5x+2
dataset_gen.set_coefficient([5, 2])

x_data, y_data = dataset_gen.make_dataset()

x_mean = np.mean(x_data)
x_std = np.std(x_data)
print(x_mean, x_std)

x_data = (x_data - x_mean)/x_std
print(np.mean(x_data), np.std(x_data))

data=np.hstack((x_data, y_data))

# model implementation
node1 = nodes.mul_node()
node2 = nodes.plus_node()

# MSE cost implementation
node3 = nodes.minus_node()
node4 = nodes.square_node()

th1, th0 = 1, 0
lr = 0.01
epochs = 2


th1_list, th0_list = [], []
loss_list = []

for epoch in range(epochs):
    for data_idx, (x, y) in enumerate(data):
        z1 = node1.forward(th1, x)
        z2 = node2.forward(z1, th0)
        z3 = node3.forward(y, z2)
        l = node4.forward(z3)

        dz3 = node4.backward(1)
        dy, dz2 = node3.backward(dz3)
        dz1, dth0 = node2.backward(dz2)
        dth1, dx = node1.backward(dz1)

        th1 -= lr * dth1
        th0 -= lr * dth0

        th1_list.append(th1)
        th0_list.append(th0)
        loss_list.append(l)

plt_show(th1_list, th0_list, loss_list, x_data, y_data)