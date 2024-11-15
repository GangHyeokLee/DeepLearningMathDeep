import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class dataset_generator:
    def __init__(self, feature_dim=1, n_sample=100, noise=0):
        self._feature_dim = feature_dim
        self._n_sample = n_sample
        self._noise = noise

        self._coefficient = None
        self._init_set_coefficient()

    def _init_set_coefficient(self):
        self._coefficient = [1 for _ in range(self._feature_dim)] + [0]

    def set_n_sample(self, n_sample):
        self._n_sample = n_sample

    def set_noise(self, noise):
        self._noise = noise

    def set_coefficient(self, coefficient_list):
        self._coefficient = coefficient_list

    def make_dataset(self):
        x_data=np.random.normal(0, 1, size=(self._n_sample, self._feature_dim))
        y_data = np.zeros(shape=(self._n_sample, 1))
        for feature_idx in range(self._feature_dim):
          y_data += self._coefficient[feature_idx] * x_data[:, feature_idx].reshape(-1, 1)
        y_data += self._coefficient[-1]
        y_data += self._noise * np.random.normal(0,1, size = (self._n_sample, 1))

        return x_data, y_data
    def dataset_visualizer(self):
        if self._feature_dim == 1:
            sns.set_theme()

            fig, ax = plt.subplots(figsize = (10, 10))
            ax.plot(x_data, y_data, 'bo',
                    alpha=0.3, markersize=20)
            ax.tick_params(axis = 'both', labelsize = 30)
            ax.set_title("DATASET", fontsize=40, color = 'darkred')
            ax.set_xlabel("X data", fontsize=30, alpha=0.6)
            ax.set_ylabel("Y data", fontsize=30, alpha=0.6)
            plt.show()
        else:
            class feature_dim_error(Exception):
                pass
            raise feature_dim_error("Visualization is valid for only feature_dim == 1")

data_gen = dataset_generator(noise=0.3)
print(data_gen._feature_dim)
print(data_gen._coefficient)

data_gen.set_coefficient([2, -1])
print(data_gen._coefficient)

x_data, y_data = data_gen.make_dataset()

data_gen.dataset_visualizer()

