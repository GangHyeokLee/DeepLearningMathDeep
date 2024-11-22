import numpy as np
import matplotlib.pyplot as plt

from LinearRegression.util.LR_dataset_generator import LR_dataset_generator as dataset_generator
from LinearRegression.util import basic_node as nodes

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

node1 = [None] + [nodes.mul_node() for _ in range(feature_dim)]
node2 = [None] + [nodes.plus_node() for _ in range(feature_dim)]

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
        Z1 = [None] * (feature_dim+1)
        Z2 = [None] * (feature_dim+1)

        for node_idx in range(1, feature_dim+1):
            Z1[node_idx] = node1[node_idx].forward(Th[node_idx], X[:, node_idx])
        Z2[1] = node2[1].forward(Th[0], Z1[1])
        for node_idx in range(2, feature_dim+1):
            Z2[node_idx] = node2[node_idx].forward(Z2[node_idx-1], Z1[node_idx])

        Z3 = node3.forward(Y, Z2[-1])
        Z4 = node4.forward(Z3)
        J = node5.forward(Z4)

        dZ1 = [None] * (feature_dim+1)
        dZ2 = [None] * (feature_dim+1)
        dTh = [None] * (feature_dim+1)

        dz4= node5.backward(1)
        dz3 = node4.backward(dz4)
        _, dZ2[-1] = node3.backward(dz3)

        for node_idx in reversed(range(1, feature_dim+1)):
            dZ2[node_idx-1], dZ1[node_idx] = node2[node_idx].backward(dZ2[node_idx])

        dTh[0] = dZ2[0]

        for node_idx in reversed(range(1, feature_dim+1)):
            dTh[node_idx], _ = node1[node_idx].backward(dZ1[node_idx])

        for th_idx in range(1, feature_dim+1):
            Th[th_idx] -= lr * np.sum(dTh[th_idx])

        th_current = Th.reshape(-1, 1)
        th_accum = np.hstack((th_accum, th_current))
        cost_list.append(J)