"""
See if we can advect on the earth surface!
"""
import numpy as np
import matplotlib.pyplot as plt

import generate_field
from openCL_driver import openCL_advect
from plot_advection import plot_advection, plot_ocean_advection


def test_hycom():
    field = generate_field.hycom_surface()
    field.V = field.V[0]  # just one timestep
    field.U = field.U[0]
    land = np.isnan(field.U)
    field.U[land] = 0
    field.V[land] = 0
    field.time = np.zeros(1)

    # initialize particles
    [X, Y] = np.meshgrid(field.x, field.y)
    ocean_points = np.array([X[~land.T], Y[~land.T]]).T
    num_particles = 5000
    p0 = ocean_points[np.random.choice(np.arange(len(ocean_points)), size=num_particles, replace=False)]

    # initialize advection parameters
    num_timesteps = 2 * 7 * 52 * 1  # 1 years
    save_every = 7 * 2  # 2 weeks
    dt = 3600 * 12  # 12 hrs
    device_index = 2  # amd
    P, buf_time, kernel_time = openCL_advect(field, p0, num_timesteps, save_every, dt,
                                             device_index, verbose=True, kernel='lat_lon')
    P = np.concatenate([p0[:, np.newaxis], P], axis=1)

    plot_ocean_advection(P, np.arange(num_timesteps, step=save_every)/2)
    return P


def test_ocean():
    field = generate_field.multiple_timestep_ocean()

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

    plot_advection(P, dt*np.arange(num_timesteps, step=save_every), field)
    return P

#P = test_hycom()
P = test_ocean()

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