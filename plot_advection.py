import matplotlib.pyplot as plt
import numpy as np


def plot_advection(P, time, field):
    # plot le advection
    fig, ax = plt.subplots(1, 1)
    t_idx = -1
    dot, = ax.plot(P[:, 0, 0], P[:, 0, 1], '.', markersize=5)

    for i in range(len(time)):
        new_t_idx = np.argmin(np.abs(field.time - time[i]))
        if new_t_idx != t_idx:
            t_idx = new_t_idx
            ax.clear()
            dot, = ax.plot(P[:, i, 0], P[:, i, 1], '.', markersize=5)
            ax.streamplot(field.x, field.y, field.U[t_idx].T, field.V[t_idx].T)
        dot.set_xdata(P[:, i, 0])
        dot.set_ydata(P[:, i, 1])
        ax.set_title('t={:.2f}'.format(time[i]))
        plt.pause(.1)
