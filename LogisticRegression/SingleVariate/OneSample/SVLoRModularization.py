import numpy as np
from SVLoR import SVLoR
from SVLoRModules import get_xdict, BinaryCrossEntropy_Loss, result_tracker, result_visualizer
from LogisticRegression.SingleVariate.OneSample.dataset_generator import dataset_generator

x_dict = get_xdict(0, 0.3, 500, 0, 0, 1)

# data = np.array([[0, -0.2, 0], [0, 0.3, 1]])
data = dataset_generator(x_dict)

model = SVLoR()
BCE_loss = BinaryCrossEntropy_Loss()

th_accum = model.get_Th()

loss_list = []
iter_idx, check_freq = 0, 2
epochs, lr = 300, 0.01

for epoch in range(epochs):
    np.random.shuffle(data)

    for data_idx in range(data.shape[0]):
        x, y = data[data_idx, 1], data[data_idx, -1],

        pred = model.forward(x)
        loss = BCE_loss.forward(y, pred)

        dpred = BCE_loss.backward()
        model.backward(dpred, lr)

        iter_idx, th_accum = result_tracker(iter_idx, check_freq, th_accum, model, loss_list, loss)

result_visualizer(th_accum, loss_list, data)