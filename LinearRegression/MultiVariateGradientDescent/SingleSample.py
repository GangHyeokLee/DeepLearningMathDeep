import numpy as np
import matplotlib.pyplot as plt

from LinearRegression.util.LR_dataset_generator import LR_dataset_generator
from LinearRegression.util import basic_node as nodes

plt.style.use("seaborn-v0_8-whitegrid")
np.random.seed(0)

dataset_gen = LR_dataset_generator(feature_dim=2, n_sample=100)
dataset_gen.set_coefficient([3, 3, 3])

distribution_params = {
    1: {'mean': 0, 'std': 1},
    2: {'mean': 0, 'std': 1},
}

dataset_gen.set_distribution_params(distribution_params)
data = dataset_gen.make_dataset()

### model

node1_1 = nodes.mul_node()
node1_2 = nodes.mul_node()

node2_1 = nodes.plus_node()
node2_2 = nodes.plus_node()

### Loss
node3 = nodes.minus_node()
node4 = nodes.square_node()

th2, th1, th0 = .5, .5, .5

epochs, lr = 3, 0.005

th_accum = np.array([th2, th1, th0]).reshape(-1, 1)
lost_list = []
for epoch in range(epochs):
    for data_idx, (_, x2, x1, y) in enumerate(data):
        z1_1 = node1_1.forward(th1, x1)
        z1_2 = node1_2.forward(th2, x2)

        z2_1 = node2_1.forward(th0, z1_1)
        z2_2 = node2_2.forward(z2_1, z1_2)

        z3 = node3.forward(y, z2_2)
        l = node4.forward(z3)

        dz3 = node4.backward(1)
        dy, dz2_2 = node3.backward(dz3)
        dz2_1, dz1_2 = node2_2.backward(dz2_2)
        dth0, dz1_1 = node2_1.backward(dz2_1)
        dth2, dx2 = node1_2.backward(dz1_2)
        dth1, dx1 = node1_1.backward(dz1_1)

        th2 = th2 - lr * dth2
        th1 = th1 - lr * dth1
        th0 = th0 - lr * dth0

        th_current = np.array([th2, th1, th0]).reshape(-1, 1)
        th_accum = np.hstack((th_accum, th_current))

        lost_list.append(l)

fig, ax = plt.subplots(2, 1, figsize=(30, 15))

ax[0].plot(th_accum[0], label=r'$\theta_{2}$', linewidth=5)
ax[0].plot(th_accum[1], label=r'$\theta_{1}$', linewidth=5)
ax[0].plot(th_accum[2], label=r'$\theta_{0}$', linewidth=5)

ax[0].tick_params(axis='both', labelsize=30)
ax[0].set_title(r'$\theta$', fontsize=40)
ax[0].legend(loc='lower right', fontsize=30)

ax[1].plot(lost_list)
ax[1].tick_params(axis='both', labelsize=30)
ax[1].set_title('Loss', fontsize=40)

plt.show()
