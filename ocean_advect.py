"""
See if we can advect on the earth surface!
"""
from generate_field import eastward_ocean
import matplotlib.pyplot as plt
import numpy as np

from openCL_driver import openCL_advect

# initialize field
from plot_advection import plot_advection

field = eastward_ocean()

# visualize field


# initialize particles
[X, Y] = np.meshgrid(np.linspace(-180, 180, 20), np.linspace(-85, 85, 10))
p0 = np.array([X.flatten(), Y.flatten()]).T

# visualize field/initialization
#plt.figure()
#plt.streamplot(field.x, field.y, field.U[0].T, field.V[0].T)
#plt.show()
#plt.plot(p0[:, 0], p0[:, 1], '.')

# initialize advection parameters
num_timesteps = 1000
save_every = 10
dt = 3600*24  # one day
device_index = 0  # cpu
P, buf_time, kernel_time = openCL_advect(field, p0, num_timesteps, save_every, dt,
                                         device_index, verbose=True, kernel='lat_lon')

plot_advection(P, np.arange(num_timesteps, step=save_every), field)
