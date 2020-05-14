"""
See if we can advect on the earth surface!
"""
from generate_field import eastward_ocean, equator_converging_ocean
import matplotlib.pyplot as plt
import numpy as np

from openCL_driver import openCL_advect

# initialize field
from plot_advection import plot_advection

field = equator_converging_ocean() + eastward_ocean()

# visualize field


# initialize particles
[X, Y] = np.meshgrid(np.linspace(-180, 180, 47), np.linspace(-85, 85, 29))
p0 = np.array([X.flatten(), Y.flatten()]).T

# visualize field/initialization
#plt.figure()
#plt.streamplot(field.x, field.y, field.U[0].T, field.V[0].T)
#plt.show()
#plt.plot(p0[:, 0], p0[:, 1], '.')

# initialize advection parameters
num_timesteps = 100
save_every = 1
dt = 3600*1  # 1 hrs
device_index = 2  # amd
P, buf_time, kernel_time = openCL_advect(field, p0, num_timesteps, save_every, dt,
                                         device_index, verbose=True, kernel='lat_lon')

plot_advection(P, np.arange(num_timesteps, step=save_every), field)
