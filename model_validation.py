"""
Here we test the cartesian and gaussian (lat/lon) kernels against well-understood results
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.patches import Circle

import generate_field
from Field2D import Field2D
from openCL_driver import openCL_advect
from plot_advection import plot_advection


def test_cartesian_electric_dipole():
    field1 = generate_field.electric_dipole(nx=1000)
    field2 = generate_field.electric_dipole(nx=1000)

    field = Field2D(x=field1.x, y=field1.y, time=np.array([0, .5]),
                    U=np.concatenate([field1.U, -field2.U], axis=0),
                    V=np.concatenate([field1.V, -field2.V], axis=0))

    # advect particles
    np.random.seed(2)
    p0 = np.random.rand(1000, 2) * [field.x.max() - field.x.min(), field.y.max() - field.y.min()] + [field.x.min(), field.y.min()]
    num_timesteps = 600
    save_every = 10
    dt = 1e-3
    P, buffer_seconds, kernel_seconds = openCL_advect(field=field, p0=p0, t0=0, num_timesteps=num_timesteps, save_every=save_every,
                                                      dt=dt, device_index=2, verbose=True)

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect(1)
    dot, = ax.plot([], [], '.', markersize=5, color='C0')
    charge1 = ax.add_artist(Circle((-1, 0), 0.05, color='blue'))
    charge2 = ax.add_artist(Circle((1, 0), 0.05, color='red'))
    ax.streamplot(field.x, field.y, field.U[0].T, field.V[0].T, linewidth=1, density=2, arrowsize=0, color='gray')
    ax.set_xlim([min(field.x), max(field.x)])
    ax.set_ylim([min(field.y), max(field.y)])
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    time = dt*np.arange(num_timesteps, step=save_every)

    def update(i):
        if i == 0:
            charge1.set_color('blue')
            charge2.set_color('red')
        elif i == 24:
            charge1.set_color('red')
            charge2.set_color('blue')

        dot.set_xdata(P[:, i, 0])
        dot.set_ydata(P[:, i, 1])
        ax.set_title(f't={i}')
        ax.set_xlim([min(field.x), max(field.x)])
        ax.set_ylim([min(field.y), max(field.y)])
        return dot, charge1, charge2

    ani = FuncAnimation(fig, update, frames=len(time))
    plt.show()
    #print('saving animation...')
    #ani.save('electric_dipole.mp4', fps=30)

test_cartesian_electric_dipole()
