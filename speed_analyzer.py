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
    t0 = 0
    dt = 1
    device_index = {'cpu': 0, 'iris': 1, 'amd': 2}[device]
    P, buffer_seconds, kernel_seconds = openCL_advect(field, p0, t0, num_timesteps, save_every, dt, device_index, verbose)

    return P, buffer_seconds, kernel_seconds


def opencl_particle_dependence():
    num_particles = np.uint32(2**np.arange(0, 24, 2))
    num_timesteps = 100
    save_every = 50
    devices = ['cpu', 'amd']  # iris pukes after 1e6 particles
    buf_s = np.zeros([len(devices), len(num_particles)])
    kern_s = np.zeros([len(devices), len(num_particles)])
    for i, device in enumerate(devices):
        for k, particles in enumerate(num_particles):
            print(f'calculating {particles} particles on {device}')
            P, buffer_seconds, kernel_seconds = run_opencl(particles, num_timesteps, save_every, device=device)
            buf_s[i, k] = buffer_seconds
            kern_s[i, k] = kernel_seconds

    plt.figure(figsize=[11, 7])
    ax = plt.gca()
    for i in range(len(devices)):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(num_particles, buf_s[i], '--', color=color, label=f'memory ({devices[i]})')
        plt.plot(num_particles, kern_s[i], '-', color=color, label=f'kernel ({devices[i]})')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('number of particles')
    plt.ylabel('time (s)')
    plt.legend()
    plt.title(f"num_timesteps={num_timesteps}, save_every={save_every}")
    plt.savefig('plots/particles_vs_speed.png', dpi=500)


def opencl_timestep_dependence():
    num_timesteps = np.uint32(2**np.arange(1, 11, 1))
    num_particles = 100000
    save_every = 1
    devices = ['cpu', 'amd']  # iris pukes after 1e6 particles
    buf_s = np.zeros([len(devices), len(num_timesteps)])
    kern_s = np.zeros([len(devices), len(num_timesteps)])
    for i, device in enumerate(devices):
        for k, nt in enumerate(num_timesteps):
            print(f'calculating {nt} timesteps on {device}')
            P, buffer_seconds, kernel_seconds = run_opencl(num_particles=num_particles,
                                                           num_timesteps=nt,
                                                           save_every=1,
                                                           device=device,
                                                           verbose=False)
            buf_s[i, k] = buffer_seconds
            kern_s[i, k] = kernel_seconds

    fig, ax = plt.subplots(figsize=[11, 7])
    for i in range(len(devices)):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(num_timesteps, buf_s[i], '--', color=color, label=f'memory ({devices[i]})')
        plt.plot(num_timesteps, kern_s[i], '-', color=color, label=f'kernel ({devices[i]})')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('number of timesteps')
    plt.ylabel('time (s)')
    plt.legend()
    plt.title(f"num_particles={num_particles}, save_every={save_every}")
    plt.savefig('plots/timesteps_vs_speed.png', dpi=500)

opencl_timestep_dependence()
