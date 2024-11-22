import numpy as np

from LinearRegression.util.dataset_generator import dataset_generator
from SVLR import SVLR


def get_data_batch(data, batch_idx):
    if batch_idx is n_batch - 1:
        batch = data[batch_idx * batch_size:]
    else:
        batch = data[batch_idx * batch_size: (batch_idx + 1) * batch_size]

    return batch

np.random.seed(0)

##### Params Setting
# model params setting
t_th1, t_th0 = 5, 5
th1, th0 = 1, 1

# data params setting
distribution_params = {'feature_0': {'mean': 0, 'mtd': 1}}

# learning params setting
lr = 0.01
epochs = 10
batch_size = 4
check_freq = 3

##### Dataset Preparation
dataset_gen = dataset_generator()
dataset_gen.set_distriution_params(distribution_params)
x_data, y_data = dataset_gen.make_dataset()
data = np.hstack((x_data, y_data))
n_batch = np.ceil(data.shape[0] / batch_size).astype(int)

# Learning
model = SVLR(th1, th0)

for epoch in range(epochs):
    np.random.shuffle(data)

    for batch_idx in range(n_batch):
        batch = get_data_batch(data, batch_idx)

        model.forward(batch)
        model.backward(lr)

    model.result_visualization()
