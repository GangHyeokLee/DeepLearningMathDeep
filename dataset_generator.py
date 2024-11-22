import numpy as np

np.random.seed(0)

def dataset_generator(x_dict):
    x_data = np.random.normal(x_dict['mean'], x_dict['std'],x_dict['n_sample'])
    x_data_noise = x_data + x_dict['noise_factor'] * np.random.normal(0, 1, x_dict['n_sample'])

    if x_dict['direction'] > 0:
        y_data = (x_data_noise > x_dict['cutoff']).astype(int)
    else:
        y_data = (x_data_noise < x_dict['cutoff']).astype(int)

    data = np.zeros(shape=(x_dict['n_sample'], 1))
    data = np.hstack((data, x_data.reshape(-1, 1), y_data.reshape(-1, 1)))

    return data