import numpy as np

class dataset_generator:
    def __init__(self, feature_dim, n_sample=300, noise_factor=0., direction=1):
        self._feature_dim = feature_dim
        self._n_sample = n_sample
        self._noise_factor = noise_factor
        self._direction = direction

        self._init_feature_dict()
        self._init_t_th()

    def _init_feature_dict(self):
        self._feature_dict = dict()
        for feature_idx in range(1, self._feature_dim + 1):
            x_dict = {'mean': 0, 'std': 1}
            self._feature_dict[feature_idx] = x_dict

    def _init_t_th(self):
        self._t_th = [0] + [1 for i in range(self._feature_dim)]

    def set_feature_dict(self, feature_dict):
        if len(feature_dict) != self._feature_dim:
            class FeatureDictError(Exception):
                pass

            raise FeatureDictError('The length of "feature_dict" should be equal to "feature_dim"')
        else:
            self._feature_dict = feature_dict

    def set_t_th(self, t_th_list):
        if len(t_th_list) != len(self._t_th):
            class t_th_Error(Exception):
                pass

            raise t_th_Error('The length of "t_th_list" should be equal to "feature_dim + 1"')
        else:
            self._t_th = t_th_list

    def make_dataset(self):
        x_data = np.zeros(shape=(self._n_sample, 1))
        y = np.zeros(shape=(self._n_sample, 1))

        for feature_idx in range(1, self._feature_dim + 1):
            feature_dict = self._feature_dict[feature_idx]
            data = np.random.normal(loc=feature_dict['mean'], scale=feature_dict['std'], size=(self._n_sample, 1))

            x_data = np.hstack((x_data, data))
            y += self._t_th[feature_idx] * data

        y += self._t_th[0]
        y_noise = y + self._noise_factor * np.random.normal(0, 1, size=(self._n_sample, 1))
        if self._direction > 0:
            y_data = (y_noise > 0).astype(int)
        else:
            y_data = (y_noise < 0).astype(int)

        data = np.hstack((x_data, y_data))

        return data