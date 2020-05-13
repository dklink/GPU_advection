import numpy as np
from Field2D import Field2D


def converge_diverge():
    # generate a field
    time = np.array([0, 50])  # seconds
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

    return Field2D(time, x, y, U * .01, V * .01)
