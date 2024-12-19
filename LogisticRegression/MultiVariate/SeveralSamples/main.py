import numpy as np

from dataset import *
from functions import *
import matplotlib as mpl
import matplotlib.pyplot as plt

np.random.seed(0)
plt.style.use('ggplot')
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 30
mpl.rcParams['axes.titlesize'] = 40
mpl.rcParams['legend.fontsize'] = 30

feature_dim = 2
n_samples = 1000
noise_factor = 0.1

model = MVLoR(feature_dim=2)
dataset_gen = dataset_generator(feature_dim=feature_dim, n_sample=n_samples, noise_factor=noise_factor)

data = dataset_gen.make_dataset()

print(data.shape)

batch_size = 32
n_batch = np.ceil(data.shape[0] / batch_size).astype(int)

th_accum = model.get_Th()
cost_list = []
epochs, lr = 50, 0.05

iter_idx, check_freq = 0, 5

BCE_cost = BinaryCrossEntropy_Cost()
for epochs in range(epochs):
    np.random.shuffle(data)

    for batch_idx in range(n_batch):
        batch = get_data_batch(data, batch_idx, n_batch, batch_size)

        X, Y = batch[:, :-1], batch[:, -1]

        Pred = model.forward(X)
        J = BCE_cost.forward(Y, Pred)

        dPred = BCE_cost.backward()
        model.backward(dPred, lr)

        iter_idx, th_accum = result_tracker(iter_idx, check_freq, th_accum, model, cost_list, J)

result_visualizer(th_accum, feature_dim)
plot_classifier_with_projection(data, th_accum, model._sigmoid)