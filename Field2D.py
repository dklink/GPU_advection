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
