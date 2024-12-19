from dataset import *
from functions import *

feature_dim = 2
n_samples = 1280
noise_factor = 0.00
size = 10

triangle = triangle_dataset_generator(n_sample=n_samples, size=size, noise=noise_factor)

data = triangle.make_dataset()
triangle.plot_dataset(data)

model = MVLoR(feature_dim=2, neuron_type="bottom")

batch_size = 128
n_batch = np.ceil(data.shape[0] / batch_size).astype(np.int64)

th_accum = model.get_Th()
cost_list = []

epochs, lr = 2000, 0.001

iter_idx, check_freq = 0, 5

BCE_cost= BinaryCrossEntropy_Cost()

for epoch in range(epochs):
    np.random.shuffle(data)

    for batch_idx in range(n_batch):
        batch = get_data_batch(data, batch_idx, n_batch, batch_size)

        X, Y = batch[:, :-1], batch[:, -1]

        Pred = model.forward(X)
        J = BCE_cost.forward(Y, Pred)

        dPred = BCE_cost.backward()
        model.backward(dPred, lr)

        iter_idx, th_accum = result_tracker(iter_idx, check_freq, th_accum, model, cost_list, J)

# result_visualizer(th_accum, feature_dim)
plot_classifier_with_projection(data, th_accum, model._sigmoid)