from functions import *
from dataset import *

feature_dim = 2
noise_factor = 0.
direction = 1
n_sample = 100

x_dict = {1: {'mean': 0, 'std': 2},
          2: {'mean': 0, 'std': 1}}

t_th_list = [0, 1, 1]

epochs, lr = 100, 0.01
iter_idx, check_freq = 0, 1

data_gen = dataset_generator(feature_dim=feature_dim, n_sample=n_sample, noise_factor=noise_factor, direction=direction)

data_gen.set_t_th(t_th_list)
data_gen.set_feature_dict(x_dict)

data = data_gen.make_dataset()

affine = Affine_Function(feature_dim=feature_dim)
sigmoid = Sigmoid()
BCE_loss = BinaryCrossEntropy_Loss()

loss_list = []
th_accum = affine.get_Th()

for epoch in range(epochs):
    np.random.shuffle(data)
    for data_idx in range(data.shape[0]):
        x, y = data[data_idx, :-1], data[data_idx, -1],

        z = affine.forward(x)
        pred = sigmoid.forward(z)
        l = BCE_loss.forward(y, pred)

        dpred = BCE_loss.backward()
        dz = sigmoid.backward(dpred)
        affine.backward(dz, lr)

        iter_idx, th_accum = result_tracker(iter_idx, check_freq, th_accum, affine, loss_list, l)

visualize_all(data, th_accum, sigmoid, feature_dim)