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

batch_size = 100

n_sample = 100
feature_dim = 3
coefficient_list = [3, 3, 3, 3]

distribution_params = {
    1: {'mean': 0, 'std': 1},
    2: {'mean': 0, 'std': 1},
    3: {'mean': 0, 'std': 1}
}

y_data = np.zeros(shape=(n_sample, 1))
x_data = np.zeros(shape=(n_sample, 1))

for feature_idx in range(1, feature_dim + 1):
    feature_data = np.random.normal(loc=distribution_params[feature_idx]['mean'],
                                    scale=distribution_params[feature_idx]['std'],
                                    size=(n_sample, 1))
    x_data = np.hstack((x_data, feature_data))

    y_data += coefficient_list[feature_idx] * feature_data

y_data += coefficient_list[0]

data = np.hstack((x_data, y_data))
n_batch = np.ceil(data.shape[0] / batch_size).astype(int)
Th = np.random.normal(0, 1, size=(feature_dim+1)).reshape(-1, 1)
epochs, lr = 200, 0.001

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

        for th_idx in range(len(dTh)):
            Th[th_idx] -= lr * np.sum(dTh[th_idx])

        th_current = Th.reshape(-1, 1)
        th_accum = np.hstack((th_accum, th_current))
        cost_list.append(J)

fig, ax = plt.subplots(2, 1, figsize=(40, 20))
for i in range(feature_dim + 1):
    ax[0].plot(th_accum[i], label=r'$\theta_{%d}$' % i, linewidth=5)

ax[1].plot(cost_list)
ax[0].legend(loc='lower right',
             fontsize=30)
ax[0].tick_params(axis='both', labelsize=30)
ax[1].tick_params(axis='both', labelsize=30)
ax[0].set_title(r'$\vec{\theta}$', fontsize=40)
ax[1].set_title('Loss', fontsize=40)

plt.show()
