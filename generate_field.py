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


def hycom_surface():
    ds = xr.open_dataset('./data/hycom_formatted.nc')

    return Field2D(ds.time.data, ds.x.data, ds.y.data, ds.water_u.sel(z=0).data.swapaxes(1, 2), ds.water_v.sel(z=0).data.swapaxes(1, 2))
