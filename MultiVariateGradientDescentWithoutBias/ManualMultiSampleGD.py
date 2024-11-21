from util.dataset_generator import *
from util import basic_node as nodes

plt.style.use("ggplot")
np.random.seed(0)

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

node1 = [None] + [nodes.mul_node() for _ in range(feature_dim)]
node2 = [None] + [nodes.plus_node() for _ in range(feature_dim)]
node3 = nodes.minus_node()
node4 = nodes.square_node()

th_list = [0.5, 0.5, .5, .5]
epochs, lr = 5, 0.001

th_accum = np.array(th_list).reshape(-1, 1)
loss_list = []

for epoch in range(epochs):
    for data_idx, (X, y) in enumerate(zip(x_data, y_data)):
        z1_1 = node1[1].forward(th_list[1], X[1])
        z1_2 = node1[2].forward(th_list[2], X[2])
        z1_3 = node1[3].forward(th_list[3], X[3])

        z2_1 = node2[1].forward(th_list[0], z1_1)
        z2_2 = node2[2].forward(z2_1, z1_2)
        z2_3 = node2[3].forward(z2_2, z1_3)

        z3 = node3.forward(y, z2_3)
        l = node4.forward(z3)

        dz3 = node4.backward(1)
        dy, dz2_3 = node3.backward(dz3)
        dz2_2, dz1_3 = node2[3].backward(dz2_3)
        dz2_1, dz1_2 = node2[2].backward(dz2_2)
        dth0, dz1_1 = node2[1].backward(dz2_1)

        dth3, dx3 = node1[3].backward(dz1_3)
        dth2, dx2 = node1[2].backward(dz1_2)
        dth1, dx1 = node1[1].backward(dz1_1)

        th_list[3] = th_list[3] - lr * dth3
        th_list[2] = th_list[2] - lr * dth2
        th_list[1] = th_list[1] - lr * dth1
        th_list[0] = th_list[0] - lr * dth0

        loss_list.append(l)

        th_next = np.array(th_list).reshape(-1, 1)
        th_accum = np.hstack((th_accum, th_next))

fig, ax = plt.subplots(2, 1, figsize=(40, 20))
for i in range(feature_dim + 1):
    ax[0].plot(th_accum[i], label=r'$\theta_{%d}$' % i, linewidth=5)

ax[1].plot(loss_list)
ax[0].legend(loc='lower right',
             fontsize=30)
ax[0].tick_params(axis='both', labelsize=30)
ax[1].tick_params(axis='both', labelsize=30)
ax[0].set_title(r'$\vec{\theta}$', fontsize=40)
ax[1].set_title('Loss', fontsize=40)

plt.show()