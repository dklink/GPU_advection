"""
This file houses scripts to test the speed of various advection algorithms,
across different problem sizes, and different devices.
"""
import generate_field
from openCL_driver import openCL_advect
import numpy as np
import matplotlib.pyplot as plt

from plot_advection import plot_advection


def run_opencl(num_particles, num_timesteps, save_every=1, device='amd', verbose=False):
    field = generate_field.converge()
    p0 = np.random.rand(num_particles, 2) * [field.x.max() - field.x.min(), field.y.max() - field.y.min()] + [field.x.min(), field.y.min()]
    dt = 1
    device_index = {'cpu': 0, 'iris': 1, 'amd': 2}[device]
    P, buffer_seconds, kernel_seconds = openCL_advect(field, p0, num_timesteps, save_every, dt, device_index, verbose)

    return P, buffer_seconds, kernel_seconds


def opencl_particle_dependence():
    num_particles = np.uint32(2**np.arange(0, 24, 2))
    devices = ['cpu', 'amd']  # iris pukes after 1e6 particles
    buf_s = np.zeros([len(devices), len(num_particles)])
    kern_s = np.zeros([len(devices), len(num_particles)])
    for i, device in enumerate(devices):
        for k, particles in enumerate(num_particles):
            print(f'calculating {particles} particles on {device}')
            P, buffer_seconds, kernel_seconds = run_opencl(particles, 100, 50, device=device)
            buf_s[i, k] = buffer_seconds
            kern_s[i, k] = kernel_seconds

    fig, ax = plt.subplots()
    for i in range(len(devices)):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(num_particles, buf_s[i], '--', color=color, label=f'memory ({devices[i]})')
        plt.plot(num_particles, kern_s[i], '-', color=color, label=f'kernel ({devices[i]})')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('number of particles')
        plt.ylabel('time (s)')
        plt.legend()


#opencl_particle_dependence()

"""
num_timesteps = 100
save_every = 50
time = np.arange(num_timesteps, step=save_every)
P, _, _ = run_opencl(num_particles=int(1e7), num_timesteps=num_timesteps, save_every=save_every, device='iris', verbose=True)
#plot_advection(P, time, generate_field.converge())
"""