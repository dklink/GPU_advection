import numpy as np


class Field2D:
    def __init__(self, time, x, y, U, V):
        self.time = time
        self.x = x
        self.y = y
        self.U = U
        self.V = V

        assert U.shape == (len(time), len(x), len(y)), "Incorrect dimensions of U"
        assert V.shape == (len(time), len(x), len(y)), "Incorrect dimensions of V"

    def inbounds(self, particle):
        """particles: numpy array, shape [3,], containing t,x,y position
            return: boolean, specifying whether particle is in field bounds"""
        return (min(self.x) < particle[0] < max(self.x) and
                min(self.y) < particle[1] < max(self.y))

    def __add__(self, other):
        assert self.U.shape == other.U.shape, "field dimensions not compatible"
        assert np.all(self.x == other.x), "x axes not compatible"
        assert np.all(self.y == other.y), "y axes not compatible"
        assert np.all(self.time == other.time), "time axes not compatible"

        return Field2D(self.time, self.x, self.y, self.U + other.U, self.V + other.V)
