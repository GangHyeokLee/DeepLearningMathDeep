import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Parameters
n_sample = 100
feature_dim = 3
coefficient_list = [3, 3, 3, 3]
distribution_params = {
    1: {'mean': 0, 'std': 1},
    2: {'mean': 0, 'std': 1},
    3: {'mean': 0, 'std': 1}
}
th_list = [0.5, 0.5, 0.5, 0.5]
epochs = 5
lr = 0.001

# Generate synthetic data
np.random.seed(0)
X = np.random.normal(distribution_params[1]['mean'], distribution_params[1]['std'], size=(n_sample, feature_dim))
X = np.concatenate([X, np.ones((n_sample, 1))], axis=1)  # Adding bias term
coefficients = np.array(coefficient_list)
y = np.dot(X, coefficients) + np.random.normal(0, 0.1, size=n_sample)

# TensorFlow model
tf_X = tf.constant(X, dtype=tf.float32)
tf_y = tf.constant(y, dtype=tf.float32)

# Initialize parameters
theta = tf.Variable(th_list, dtype=tf.float32)
optimizer = tf.optimizers.SGD(learning_rate=lr)

# Lists to store results
th_accum = [[] for _ in range(feature_dim + 1)]
loss_list = []

# Training loop
def compute_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        predictions = tf.matmul(tf_X, tf.reshape(theta, (-1, 1)))[:, 0]
        loss = compute_loss(tf_y, predictions)
    grads = tape.gradient(loss, [theta])
    optimizer.apply_gradients(zip(grads, [theta]))

    # Record parameters and loss
    for i, val in enumerate(theta.numpy()):
        th_accum[i].append(val)
    loss_list.append(loss.numpy())

# Plot results
fig, ax = plt.subplots(2, 1, figsize=(40, 20))
for i in range(feature_dim + 1):
    ax[0].plot(th_accum[i], label=r'$\theta_{%d}$' % i, linewidth=5)
ax[1].plot(loss_list, label='Loss', linewidth=5)

ax[0].legend(loc='lower right', fontsize=30)
ax[0].tick_params(axis='both', labelsize=30)
ax[1].tick_params(axis='both', labelsize=30)

ax[0].set_title(r'$\vec{\theta}$', fontsize=40)
ax[1].set_title('Loss', fontsize=40)

plt.show()
