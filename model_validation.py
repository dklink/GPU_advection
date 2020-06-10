"""
Here we test the cartesian and gaussian (lat/lon) kernels against well-understood results
"""
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

import generate_field
from Field2D import Field2D
from openCL_driver import openCL_advect


def test_cartesian_electric_dipole():
    field1 = generate_field.electric_dipole(nx=1000)
    field2 = generate_field.electric_dipole(nx=1000)

    field = Field2D(x=field1.x, y=field1.y, time=np.array([0, .5]),
                    U=np.concatenate([field1.U, -field2.U], axis=0),
                    V=np.concatenate([field1.V, -field2.V], axis=0))

    # advect particles
    np.random.seed(2)
    p0 = np.random.rand(1000, 2) * [field.x.max() - field.x.min(), field.y.max() - field.y.min()] + [field.x.min(), field.y.min()]
    num_timesteps = 600
    save_every = 10
    dt = 1e-3
    P, buffer_seconds, kernel_seconds = openCL_advect(field=field, p0=p0, t0=0, num_timesteps=num_timesteps, save_every=save_every,
                                                      dt=dt, device_index=2, verbose=True)

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect(1)
    dot, = ax.plot([], [], '.', markersize=5, color='C0')
    charge1 = ax.add_artist(Circle((-1, 0), 0.05, color='blue'))
    charge2 = ax.add_artist(Circle((1, 0), 0.05, color='red'))
    ax.streamplot(field.x, field.y, field.U[0].T, field.V[0].T, linewidth=1, density=2, arrowsize=0, color='gray')
    ax.set_xlim([min(field.x), max(field.x)])
    ax.set_ylim([min(field.y), max(field.y)])
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    time = dt*np.arange(num_timesteps, step=save_every)

    def update(i):
        if i == 0:
            charge1.set_color('blue')
            charge2.set_color('red')
        elif i == 24:
            charge1.set_color('red')
            charge2.set_color('blue')

        dot.set_xdata(P[:, i, 0])
        dot.set_ydata(P[:, i, 1])
        ax.set_title(f't={i}')
        ax.set_xlim([min(field.x), max(field.x)])
        ax.set_ylim([min(field.y), max(field.y)])
        return dot, charge1, charge2

    ani = FuncAnimation(fig, update, frames=len(time))
    plt.show()
    #print('saving animation...')
    #ani.save('plots/electric_dipole.mp4', fps=30)


def test_cartesian_orbit():

    def advect_circles(dt, field):
        # advect particles
        p0 = np.array([[.4, 0]])
        num_timesteps = int(10/dt)
        save_every = 1

        P, buffer_seconds, kernel_seconds = openCL_advect(field=field, p0=p0, t0=0, num_timesteps=num_timesteps,
                                                          save_every=save_every, dt=dt, device_index=0, verbose=False)
        return P[0, :, 0], P[0, :, 1]

    gs = gridspec.GridSpec(3, 3)
    fig = plt.figure(figsize=(8, 8))
    axes = [plt.subplot(gs[i//3, i%3]) for i in range(9)]
    axes = iter(axes)
    dts = [.1, .01, .001]
    nxs = [5, 10, 30]
    for i, dt in enumerate(dts):
        for j, nx in enumerate(nxs):
            field = generate_field.concentric_circles(nx=nx)
            x, y = advect_circles(dt, field)
            ax = next(axes)
            ax.set_aspect(1)
            ax.quiver(field.x, field.y, field.U[0].T, field.V[0].T)
            ax.plot(x, y, '-')
            ax.plot(x[0], y[0], 'o', color='green', label='Initial Position')
            ax.plot(x[-1], y[-1], 's', color='red', label='Final Position')
            ax.set_xlim([min(field.x)*1.05, max(field.x)*1.05])
            ax.set_ylim([min(field.y)*1.05, max(field.y)*1.05])
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
            if j == 0:
                ax.set_ylabel(f'dt = {dt}', fontsize=11)
            else:
                ax.set_yticks([])
            if i == 2:
                ax.set_xlabel(f'{nx}x{nx}', fontsize=11)
            else:
                ax.set_xticks([])

    # Set big x/y labels
    fig.text(0.51, 0.025, 'Field Resolution', ha='center', va='center', fontsize=12)
    fig.text(0.025, 0.5, 'Temporal Resolution', ha='center', va='center', rotation='vertical', fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(.8, .9))
    fig.text(.52, .928, 'Effect of Resolution on Model Accuracy', ha='center', va='center', fontsize=14)
    plt.show()
    #plt.savefig('plots/resolution_v_accuracy.png', dpi=500)


def test_hycom_advection():
    print("Loading Data...")
    field = generate_field.hycom_surface(months=list(range(1, 13)))
    field.x = np.linspace(min(field.x), max(field.x), len(field.x))
    land = np.isnan(field.U[0])
    field.U[:, land] = 0
    field.V[:, land] = 0

    field.time = (field.time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')  # convert from numpy datetime to seconds since epoch

    # initialize particles
    [X, Y] = np.meshgrid(field.x, field.y)
    ocean_points = np.array([X[~land.T], Y[~land.T]]).T
    num_particles = 5000
    p0 = ocean_points[np.random.choice(np.arange(len(ocean_points)), size=num_particles, replace=False)]

    # initialize advection parameters
    t_start = field.time[0]
    t_end = field.time[-1]
    num_timesteps = 365*24
    time = np.linspace(t_start, t_end, num_timesteps)
    dt = time[1]-time[0]
    print(f'dt: {dt/3600: .1f} hours')
    save_every = 24
    device_index = 2  # amd
    P, buf_time, kernel_time = openCL_advect(field, p0, t_start, num_timesteps, save_every, dt,
                                             device_index, verbose=True, kernel='lat_lon')
    P = np.concatenate([p0[:, np.newaxis], P], axis=1)
    out_time = time[0] + dt*np.arange(num_timesteps+1, step=save_every)

    # make an animation
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=[14, 8])
    ax = plt.axes(projection=proj)
    ax.coastlines()

    dot, = ax.plot([], [], '.')

    def update(i):
        dot.set_xdata(P[:, i, 0])
        dot.set_ydata(P[:, i, 1])
        ax.set_title(datetime.utcfromtimestamp(out_time[i]))
        ax.set_ylim(-90, 90)
        return dot,

    ani = FuncAnimation(fig, update, frames=len(out_time))
    #plt.show()
    print('saving animation...')
    ani.save('plots/hycom_2015.mp4', fps=30)
