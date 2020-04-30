import numpy as np
from particle_advection import advect_particle
from generate_field import converge_diverge
from plot_advection import plot_advection
import time
import multiprocessing as mp


def parallel_advect(i):
    x0 = np.random.randint(np.min(field.x), np.max(field.x))
    y0 = np.random.randint(np.min(field.y), np.max(field.y))
    return advect_particle([x0, y0], field, advection_time)


# get a field
field = converge_diverge()

# advect le particles in ~parallel~
advection_time = np.arange(field.time[0], field.time[1], 1)
nparticles = 5000

tic = time.time()
pool = mp.Pool(4)  # number of cores (not including virtual cores)
res = pool.map(parallel_advect, range(nparticles))
pool.close()
pool.join()
P = np.array(res)
toc = time.time()

print('advected {} particles in {} seconds'.format(nparticles, toc-tic))

#plot_advection(P, advection_time, field)
