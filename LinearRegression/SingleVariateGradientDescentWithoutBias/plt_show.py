import matplotlib.pyplot as plt
import numpy as np

def plt_show(th_list, loss_list, x_data, y_data):
    fig, ax = plt.subplots(2, 1, figsize=(42, 20))
    ax[0].plot(th_list, linewidth=5)
    ax[1].plot(loss_list, linewidth=5)

    title_font = {'size': 50, 'alpha': 0.8, 'color': 'navy'}
    label_font = {'size': 50, 'alpha': 0.8}
    plt.style.use('seaborn-v0_8-darkgrid')

    ax[0].set_title(r'$\theta$', fontdict=title_font)
    ax[1].set_title("Cost", fontdict=title_font)
    ax[1].set_xlabel("Epoch", fontdict=label_font)
    # Adjust tick label sizes
    ax[0].tick_params(axis='both', labelsize=30)
    ax[1].tick_params(axis='both', labelsize=30)

    plt.show()

    N_line = len(th_list)
    cmap = plt.get_cmap('rainbow', lut=N_line)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(x_data, y_data)

    test_th = th_list[:N_line]
    x_range = np.array([np.min(x_data), np.max(x_data)])

    for line_idx in range(N_line):
        pred_line = np.array([x_range[0] * test_th[line_idx], x_range[1] * test_th[line_idx]])
        ax.plot(x_range, pred_line, color=cmap(line_idx), alpha=0.1)

    # Adjust tick label sizes for scatter plot
    ax.tick_params(axis='both', labelsize=20)

    plt.show()