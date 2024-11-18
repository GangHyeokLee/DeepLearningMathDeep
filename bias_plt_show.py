import matplotlib.pyplot as plt
import numpy as np

def plt_show(th1_list, th0_list, loss_list, x_data, y_data):
    plt.style.use('seaborn-v0_8-darkgrid')

    fig, ax = plt.subplots(2, 1, figsize=(20, 15))
    ax[0].plot(th1_list, label=r'$\theta_{1}$')
    ax[0].plot(th0_list, label=r'$\theta_{0}$')
    ax[1].plot(loss_list)

    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.1)

    ax[0].set_title(r'$\theta_{1}, \theta_{2}$', fontsize=30)
    ax[1].set_title("Cost", fontsize=30)
    # Adjust tick label sizes
    ax[0].legend(loc='lower right', fontsize=30)
    ax[0].axes.get_xaxis().set_visible(False)
    ax[0].tick_params(axis='both', labelsize=20)
    ax[1].tick_params(axis='both', labelsize=20)

    x_ticks = np.arange(0, len(th1_list) + 1, 10)
    y_ticks = np.arange(0, 6)
    ax[1].set_xticks(x_ticks)
    ax[0].set_yticks(y_ticks)

    plt.show()

    N_line = len(th1_list)
    cmap = plt.get_cmap('rainbow', lut=N_line)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(x_data, y_data)

    test_th1 = th1_list[:N_line]
    test_th0 = th0_list[:N_line]
    x_range = np.array([np.min(x_data), np.max(x_data)])

    for line_idx in range(N_line):
        pred_line = np.array([x_range[0] * test_th1[line_idx] + test_th0[line_idx], x_range[1] * test_th1[line_idx] + test_th0[line_idx]])
        ax.plot(x_range, pred_line, color=cmap(line_idx), alpha=0.1)

    # Adjust tick label sizes for scatter plot
    ax.tick_params(axis='both', labelsize=20)


    plt.show()