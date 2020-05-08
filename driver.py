import numpy as np
from particle_advection import advect_particle
from generate_field import converge_diverge
from plot_advection import plot_advection
import time

# get a field
field = converge_diverge()

# advect le particle
advection_time = np.arange(field.time[0], field.time[1], 1)
nparticles = 10000
P = np.zeros([nparticles, len(advection_time), 2])

tic = time.time()
for i in range(nparticles):
    x0 = np.random.randint(np.min(field.x), np.max(field.x))
    y0 = np.random.randint(np.min(field.y), np.max(field.y))
    P[i] = advect_particle([x0, y0], field, advection_time)
toc = time.time()
print('advected {} particles in {} seconds'.format(nparticles, toc-tic))

plot_advection(P, advection_time, field)
