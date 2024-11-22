import numpy as np

from LinearRegression.util import basic_node as nodes
from LinearRegression.util.dataset_generator import dataset_generator
from bias_plt_show import plt_show

np.random.seed(0)

##### Parameter Setting
t_th1, t_th0 = 5, 5
th1, th0 = 1, 1
lr = 0.01
epochs = 20
batch_size = 8

dataset_gen = dataset_generator()
dataset_gen.set_coefficient([t_th1, t_th0])
x_data, y_data = dataset_gen.make_dataset()
dataset_gen.dataset_visualizer(x_data, y_data)
data = np.hstack((x_data, y_data))
n_batch = int(data.shape[0] / batch_size)

# model implementation
node1 = nodes.mul_node()
node2 = nodes.plus_node()

# MSE cost implementation
node3 = nodes.minus_node()
node4 = nodes.square_node()
node5 = nodes.mean_node()

th1_list, th0_list = [], []
loss_list = []

for epoch in range(epochs):
    np.random.shuffle(data)
    for batch_idx in range(n_batch):
        batch = data[batch_size * batch_idx:batch_size * (batch_idx + 1)]

        z1 = node1.forward(th1, batch[:, 0])
        z2 = node2.forward(z1, th0)
        z3 = node3.forward(batch[:, 1], z2)
        L = node4.forward(z3)
        J = node5.forward(L)

        dL = node5.backward(1)
        dZ3 = node4.backward(dL)
        dY, dZ2 = node3.backward(dZ3)
        dZ1, dTh0 = node2.backward(dZ2)
        dTh1, dX = node1.backward(dZ1)

        th1 -= lr * np.sum(dTh1)
        th0 -= lr * np.sum(dTh0)

        th1_list.append(th1)
        th0_list.append(th0)
        loss_list.append(J)

plt_show(th1_list, th0_list, loss_list, x_data, y_data)
