"""
See if we can advect on the earth surface!
"""
import numpy as np
import matplotlib.pyplot as plt

from generate_field import eastward_ocean, equator_converging_ocean, hycom_surface
from openCL_driver import openCL_advect
from plot_advection import plot_advection, plot_ocean_advection


def test_hycom():
    field = hycom_surface()
    field.V = field.V[0]  # just one timestep
    field.U = field.U[0]
    field.time = np.zeros(1)

    # initialize particles
    [X, Y] = np.meshgrid(np.linspace(-180, 180, 100), np.linspace(-85, 85, 50))
    p0 = np.array([X.flatten(), Y.flatten()]).T

    # initialize advection parameters
    num_timesteps = 2 * 7 * 52 * 1  # 1 years
    save_every = 14  # 1 week
    dt = 3600 * 12  # 12 hrs
    device_index = 2  # amd
    P, buf_time, kernel_time = openCL_advect(field, p0, num_timesteps, save_every, dt,
                                             device_index, verbose=True, kernel='lat_lon')

    plot_ocean_advection(P, np.arange(num_timesteps, step=save_every))
    return P


def test_ocean():
    field = equator_converging_ocean() + eastward_ocean()  # or just one of them

    # initialize particles
    [X, Y] = np.meshgrid(np.linspace(-180, 180, 47), np.linspace(-85, 85, 29))
    p0 = np.array([X.flatten(), Y.flatten()]).T

    # initialize advection parameters
    num_timesteps = 100
    save_every = 1
    dt = 3600*1  # 1 hrs
    device_index = 2  # amd
    P, buf_time, kernel_time = openCL_advect(field, p0, num_timesteps, save_every, dt,
                                             device_index, verbose=True, kernel='lat_lon')

    plot_advection(P, np.arange(num_timesteps, step=save_every), field)


P = test_hycom()
'''plt.figure()
min, max = np.nanmin(P[:, :, 0]), np.nanmax(P[:, :, 0])
for i in range(364):
    plt.clf()
    plt.hist(P[:, i, 0])
    plt.xlim([min, max])
    plt.ylim([0, 500])
    plt.title(i)
    plt.pause(.001)
'''