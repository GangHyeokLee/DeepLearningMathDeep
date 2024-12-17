import matplotlib as mpl

from dataset import *
from functions import *

np.random.seed(0)
plt.style.use('ggplot')
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 30
mpl.rcParams['axes.titlesize'] = 40
mpl.rcParams['legend.fontsize'] = 30

model = SVLoR()

x_dict = get_xdict(1, 1, 100, 0, 0, 1)
data = dataset_generator(x_dict)

batch_size = 2
n_batch = np.ceil(data.shape[0] / batch_size).astype(int)

th_accum = model.get_Th()
cost_list = []
epochs, lr = 50, 0.05

iter_idx, check_freq = 0, 5

BCE_cost = BinaryCrossEntropy_Cost()

for epoch in range(epochs):
    np.random.shuffle(data)

    for batch_idx in range(n_batch):
        batch = get_data_batch(data, batch_idx, n_batch, batch_size)

        X, Y = batch[:, 1], batch[:, -1]

        Pred = model.forward(X)
        J = BCE_cost.forward(Y, Pred)

        dPred = BCE_cost.backward()
        model.backward(dPred, lr)

        iter_idx, th_accum = result_tracker(iter_idx, check_freq, th_accum, model, cost_list, J)

result_visualizer(th_accum, cost_list, data, x_dict.get('mean'), x_dict.get('std'), x_dict.get('n_sample'), x_dict.get('noise_factor'), batch_size, epochs, lr)
