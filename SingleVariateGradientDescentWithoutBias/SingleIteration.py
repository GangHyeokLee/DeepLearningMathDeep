import matplotlib.pyplot as plt
import numpy as np
from sympy.physics.control.control_plots import matplotlib

from dataset_generator import dataset_generator
import basic_node as nodes

dataset_gen = dataset_generator()

dataset_gen.set_coefficient([5, 0])

x_data, y_data = dataset_gen.make_dataset()
# print(x_data)
# print(y_data)

# model implementation

node1 = nodes.mul_node()

# sequare error loss implementation
node2 = nodes.minus_node()
node3 = nodes.square_node()

# hyperparameter setting
epochs = 5  # total epoch setting

lr = 0.01
th = -1

loss_list = []
th_list = []

for epoch in range(epochs):
    # 보통 랜덤 셔플 해서 데이터 입력이 랜덤이 되도록 하는데 오늘은 패스
    for data_idx in range(len(x_data)):
        x, y = x_data[data_idx], y_data[data_idx]
        z1 = node1.forward(th, x)
        z2 = node2.forward(y, z1)
        l = node3.forward(z2)

        # 첫번째 loss에 대한 제곱오차
        dz2 = node3.backward(1)
        dy, dz1 = node2.backward(dz2)
        dth, dx = node1.backward(dz1)

        th = th - lr * dth
        th_list.append(th)
        loss_list.append(z2)

fig, ax = plt.subplots(2, 1, figsize=(30, 10))
ax[0].plot(th_list)
ax[1].plot(loss_list)

title_font = {'size': 30, 'alpha': 0.8, 'color': 'navy'}
label_font = {'size': 20, 'alpha': 0.8}
plt.style.use('seaborn-v0_8-darkgrid')

ax[0].set_title(r'$\theta$', fontdict=title_font)
ax[1].set_title("Loss", fontdict=title_font)
ax[1].set_xlabel("Iteration", fontdict=label_font)

plt.show()

N_line = 200
cmap = plt.get_cmap('rainbow', lut=N_line)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(x_data, y_data)

test_th = th_list[:N_line]
x_range = np.array([np.min(x_data), np.max(x_data)])

for line_idx in range(N_line):
    pred_line = np.array([x_range[0] * test_th[line_idx], x_range[1] * test_th[line_idx]])
    ax.plot(x_range, pred_line, color=cmap(line_idx), alpha=0.1)

plt.show()