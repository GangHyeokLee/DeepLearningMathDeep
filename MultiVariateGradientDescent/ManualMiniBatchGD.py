import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from util.LR_dataset_generator import LR_dataset_generator as dataset_generator
from util import basic_node as nodes

np.random.seed(0)
plt.style.use("ggplot")

def get_data_batch(data, batch_idx):
    if batch_idx == 0:
        batch = data[batch_idx * batch_size:]
    else:
        batch = data[batch_idx * batch_size : (batch_idx+1) * batch_size]
    return batch

feature_dim = 2
batch_size = 8

data_gen = dataset_generator(feature_dim=feature_dim)
x_data, y_data = data_gen.make_dataset()
data = np.hstack((x_data, y_data))

n_batch = np.ceil(data.shape[0] / batch_size).astype(int)
Th = np.random.normal(0, 1, size=(feature_dim+1)).reshape(-1, 1)
epochs, lr = 100, 0.001

node1_1 = nodes.mul_node()
node1_2 = nodes.mul_node()

node2_1 = nodes.plus_node()
node2_2 = nodes.plus_node()

node3 = nodes.minus_node()
node4 = nodes.square_node()
node5 = nodes.mean_node()

cost_list = []
th_accum = Th.reshape(-1, 1)

for epoch in range(epochs):
    np.random.shuffle(data)

    for batch_idx in range(n_batch):
        batch = get_data_batch(data, batch_idx)
        X, Y = batch[:,:-1], batch[:,-1]

        Z1_1 = node1_1.forward(Th[1], X[:, 1])
        Z1_2 = node1_2.forward(Th[2], X[:, 2])
        Z2_1 = node2_1.forward(Th[0], Z1_1)
        Z2_2 = node2_2.forward(Z2_1. Z1_2)

        Z3 = node3.forward(Y, Z2_2)
        Z4 = node4.forward(Z3)
        J = node5.forward(Z4)

        dz4= node5.backward(1)
        dz3 = node4.backward(dz4)
        _, dz2_2 = node3.backward(dz3)

        dz2_1, dZ1_2 = node2_2.backward(dz2_2)
        dTh0, dZ1_1 = node2_1.backward(dz2_1)
        dTh2, _ = node1_2.backward(dZ1_2)
        dTh1, _ = node1_1.backward(dZ1_1)

        Th[2] -= lr * np.sum(dTh2)
        Th[1] -= lr * np.sum(dTh1)
        Th[0] -= lr * np.sum(dTh0)

        th_current = Th.reshape(-1, 1)
        th_accum = np.hstack((th_accum, th_current))
        cost_list.append(J)