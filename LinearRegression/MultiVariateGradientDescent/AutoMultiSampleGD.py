from LinearRegression.util.dataset_generator import *
from LinearRegression.util import basic_node as nodes

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
        z1_list = [None] * (feature_dim + 1)
        z2_list, dz2_list, dz1_list, dth_list = z1_list.copy(), z1_list.copy(), z1_list.copy(), z1_list.copy()

        for node_idx in range(1, feature_dim + 1):
            z1_list[node_idx] = node1[node_idx].forward(th_list[node_idx], X[node_idx])

        z2_list[1] = node2[1].forward(th_list[0], z1_list[1])

        for node_idx in range(2, feature_dim + 1):
            z2_list[node_idx] = node2[node_idx].forward(z2_list[node_idx - 1], z1_list[node_idx])

        z3 = node3.forward(y, z2_list[-1])
        l = node4.forward(z3)

        dz3 = node4.backward(1)
        _, dz2_list[-1] = node3.backward(dz3)
        for node_idx in reversed(range(1, feature_dim + 1)):
            dz2_list[node_idx-1], dz1_list[node_idx] = node2[node_idx].backward(dz2_list[node_idx])

        dth_list[0] = dz2_list[0]

        for node_idx in reversed(range(1, feature_dim + 1)):
            dth_list[node_idx], _ = node1[node_idx].backward(dz1_list[node_idx])

        for th_idx in range(len(th_list)) :
            th_list[th_idx] -= lr * dth_list[th_idx]

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
