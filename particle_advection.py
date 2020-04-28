# straightforward, single-particle advection

import numpy as np
from Field2D import Field2D


def advect_particle(p0, field: Field2D, time):
    """advect a particle with initial position p0 through field along vector time"""
    ntimesteps = len(time)
    P = np.zeros([ntimesteps, 2])  # particle position in time, space

    # initialize particle
    P[0] = p0
    for i in range(ntimesteps-1):
        # find nearest field U and V indices
        nearest_time_idx = np.argmin(np.abs(field.time - time[i]))
        nearest_x_idx = np.argmin(np.abs(field.x - P[i, 0]))
        nearest_y_idx = np.argmin(np.abs(field.y - P[i, 1]))

        dt = time[i+1] - time[i]
        dx, dy = 0, 0
        if field.inbounds(P[i]):
            dx = field.U[nearest_time_idx, nearest_x_idx, nearest_y_idx] * dt
            dy = field.V[nearest_time_idx, nearest_x_idx, nearest_y_idx] * dt
            if dy == 0:
                print('huh')
        P[i+1] = P[i] + np.array([dx, dy])

    return P
