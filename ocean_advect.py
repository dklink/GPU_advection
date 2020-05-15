"""
See if we can advect on the earth surface!
"""
import matplotlib.pyplot as plt
import numpy as np

from generate_field import eastward_ocean, equator_converging_ocean, hycom_surface
from openCL_driver import openCL_advect
from plot_advection import plot_advection


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

    plot_advection(P, np.arange(num_timesteps, step=save_every), field, streamfunc=False)


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

test_hycom()
