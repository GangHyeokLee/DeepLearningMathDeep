from dataset import *
from functions import *

feature_dim = 2
n_samples = 300
noise_factor = 0
size = 100

data_gen = LineDatasetGenerator(size, n_samples, noise_factor)
print(data_gen.shape)