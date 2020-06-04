import numpy as np
from Field2D import Field2D
import xarray as xr


def converge_diverge():
    # generate a field
    time = np.array([0, 100])  # seconds
    x = np.linspace(-100, 100, 100)
    y = np.linspace(-50, 50, 50)
    [X, Y] = np.meshgrid(x, y)
    U = np.zeros([len(time), len(x), len(y)])
    V = np.zeros([len(time), len(x), len(y)])
    U[0, :] = -X.T
    V[0, :] = -Y.T
    U[1, :] = X.T
    V[1, :] = Y.T

    return Field2D(time, x, y, U*.01, V*.01)


def converge():
    time = np.array([0])  # seconds
    x = np.linspace(-100, 100, 100)
    y = np.linspace(-50, 50, 50)
    [X, Y] = np.meshgrid(x, y)
    U = np.zeros([len(time), len(x), len(y)])
    V = np.zeros([len(time), len(x), len(y)])
    U[0, :] = -X.T
    V[0, :] = -Y.T

    return Field2D(time, x, y, U * .01, V * .01)


def eastward_ocean():
    time = np.array([0])  # seconds
    x = np.linspace(-180, 180, 1000)
    y = np.linspace(-90, 90, 500)
    U = 5 * np.ones([len(time), len(x), len(y)])
    V = np.zeros([len(time), len(x), len(y)])

    return Field2D(time, x, y, U, V)


def equator_converging_ocean():
    time = np.array([0])  # seconds
    x = np.linspace(-180, 180, 1000)
    y = np.linspace(-90, 90, 500)
    [X, Y] = np.meshgrid(x, y)
    U = np.zeros([len(time), len(x), len(y)])
    V = np.zeros([len(time), len(x), len(y)])

    V[0, :] = -np.sign(Y.T) * 5

    return Field2D(time, x, y, U, V)


def multiple_timestep_ocean():
    field1 = eastward_ocean()
    field2 = equator_converging_ocean()
    time = np.array([0, 100*3600])  # 0 hrs, 100 hrs
    U = np.concatenate([field1.U, field2.U], axis=0)
    V = np.concatenate([field1.V, field2.V], axis=0)
    return Field2D(time, field1.x, field1.y, U, V)


def hycom_surface(months):
    time = []
    x = []
    y = []
    U = []
    V = []
    for month in months:
        ds = xr.open_dataset(f'./data/uv_2015_{month}_3d.formatted.nc')
        x.append(ds.x.data)
        y.append(ds.y.data)
        time.append(ds.time.data)
        U.append(ds.water_u.sel(z=0).data.swapaxes(1, 2))
        V.append(ds.water_v.sel(z=0).data.swapaxes(1, 2))

    assert np.all(x[0] == x[i] for i in range(len(x)))
    assert np.all(y[0] == y[i] for i in range(len(x)))

    return Field2D(time=np.concatenate(time), x=x[0], y=y[0],
                   U=np.concatenate(U, axis=0), V=np.concatenate(V, axis=0))


def electric_dipole(nx=100):
    """electric field generation yoinked from https://scipython.com/blog/visualizing-a-vector-field-with-matplotlib/"""

    def E(q, r0, x, y):
        """Return the electric field vector E=(Ex,Ey) due to charge q at r0."""
        den = np.hypot(x - r0[0], y - r0[1]) ** 3
        return q * (x - r0[0]) / den, q * (y - r0[1]) / den

    # Grid of x, y points
    ny = nx
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    X, Y = np.meshgrid(x, y)

    # Create a multipole with nq charges of alternating sign, equally spaced
    # on the unit circle.
    nq = 2
    charges = []
    for i in range(nq):
        q = i % 2 * 2 - 1
        charges.append((q, (np.cos(2 * np.pi * i / nq), np.sin(2 * np.pi * i / nq))))

    # Electric field vector, E=(Ex, Ey), as separate components
    Ex, Ey = np.zeros((ny, nx)), np.zeros((ny, nx))
    for charge in charges:
        ex, ey = E(*charge, x=X, y=Y)
        Ex += ex
        Ey += ey

    return Field2D(x=x, y=y, time=np.array([0]), U=Ex.T[np.newaxis], V=Ey.T[np.newaxis])
