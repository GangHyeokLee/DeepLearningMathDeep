import numpy as np

def get_xdict(mean, std, n_sample, noise_factor, cutoff, direction):
    return {'mean': mean, 'std': std, 'n_sample': n_sample, 'noise_factor': noise_factor, 'cutoff': cutoff,
            'direction': direction}

def dataset_generator(x_dict):
    x_data = np.random.normal(x_dict['mean'], x_dict['std'], x_dict['n_sample'])
    x_data_noise = x_data + x_dict['noise_factor'] * np.random.normal(0, 1, x_dict['n_sample'])

    if x_dict['direction'] > 0:
        y_data = (x_data_noise > x_dict['cutoff']).astype(int)
    else:
        y_data = (x_data_noise < x_dict['cutoff']).astype(int)

    data = np.zeros(shape=(x_dict['n_sample'], 1))
    data = np.hstack((data, x_data.reshape(-1, 1), y_data.reshape(-1, 1)))

    return data

def get_data_batch(data, batch_idx, n_batch, batch_size):
    if batch_idx is n_batch-1:
        batch = data[batch_idx * batch_size:]
    else:
        batch = data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    return batch