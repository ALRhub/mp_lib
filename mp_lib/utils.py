from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm


def plot_mean_and_std(time, mean_traj, std_traj, indices=None):
    color_cycle = cycle(np.linspace(0, 1, 10))
    color_map = cm.get_cmap('rainbow')

    legend_handles = []

    if not indices:
        indices = range(mean_traj.shape[1])

    for i in indices:
        plt.figure()

        curve = mean_traj[:, i]
        curve_std = std_traj[:, i]

        lower_bound = curve - 2 * curve_std
        upper_bound = curve + 2 * curve_std

        x = time
        color = color_map(next(color_cycle))

        plt.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.5)

        label = 'Joint %d' % (i)
        new_handle = plt.plot(x, curve, color=color, label=label)
        legend_handles.append(new_handle[0])

    plt.legend(handles=legend_handles)

    plt.xlabel('time')
    plt.ylabel('q')

    plt.autoscale(enable=True, axis='x', tight=True)