import generate_field
import numpy as np

from openCL_driver import openCL_advect
from plot_advection import plot_advection

field = generate_field.converge_diverge()

p0 = np.random.rand(500, 2) * [field.x.max() - field.x.min(), field.y.max() - field.y.min()] + [field.x.min(), field.y.min()]
dt = 1
device_index = 0
num_timesteps = 100
save_every = 1
verbose = True

P, buffer_seconds, kernel_seconds = openCL_advect(field, p0, num_timesteps, save_every, dt, device_index, verbose)
plot_advection(P, np.arange(num_timesteps), field)
