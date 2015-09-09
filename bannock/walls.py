from __future__ import print_function, division
import numpy as np
from fealty import lattice, fields
from ciabatta.meta import make_repr_str
from bannock import maze


class Walls(fields.Field):
    """A field representing an environment filled with square obstacles."""

    def __init__(self, L, dim, dx):
        fields.Field.__init__(self, L, dim, dx)
        self.a = np.zeros(self.dim * (self.M,), dtype=np.uint8)

    @property
    def free_area_i(self):
        """Calculate the number of elements that are not occupied by obstacles.

        Returns
        -------
        area_i: int
        """
        return np.logical_not(self.a).sum()

    @property
    def free_area(self):
        """Calculate the area that is not occupied by obstacles.

        Returns
        -------
        area: float
        """
        return self.free_area_i * self.dA

    def is_obstructed(self, r):
        """Determine if a set of position vectors lie on top of obstacles.

        Parameters
        ----------
        r: array_like[shape=(n, 2)]
            Particle position vectors.

        Returns
        -------
        o: numpy.ndarray[dtype=bool, shape=(n,)]
            `True` if a vector is obstructed.
        """
        return self.a[tuple(self.r_to_i(r).T)].astype(np.bool)

    def __repr__(self):
        fs = [('L', self.L), ('dim', self.dim), ('dx', self.dx)]
        return make_repr_str(self, fs)


class HalfClosed(Walls):
    """A set of walls closing the 2D environment along one edge."""

    def __init__(self, L, dx):
        Walls.__init__(self, L, dim=2, dx=dx)
        self.a[:, 0] = True

    def __repr__(self):
        fs = [('L', self.L), ('dx', self.dx)]
        return make_repr_str(self, fs)


class Closed(Walls):
    """A set of walls closing the 2D environment at all edges."""

    def __init__(self, L, dx):
        Walls.__init__(self, L, dim=2, dx=dx)
        self.a[:, 0] = True
        self.a[:, -1] = True
        self.a[0, :] = True
        self.a[-1, :] = True

    def __repr__(self):
        fs = [('L', self.L), ('dx', self.dx)]
        return make_repr_str(self, fs)


class Tittled(Walls):

    def __init__(self, L, dx, wx, wy, sx, sy):
        Walls.__init__(self, L, dim=2, dx=dx)
        self.wx_i = int(round(wx / self.dx))
        self.wy_i = int(round(wy / self.dx))
        self.sx_i = int(round(sx / self.dx))
        self.sy_i = int(round(sy / self.dx))

        for i_x in range(self.sx_i,
                         self.a.shape[0] - self.sx_i,
                         self.sx_i):
            for i_y in range(self.sy_i,
                             self.a.shape[1] - self.sy_i,
                             self.sy_i):
                self.a[i_x - self.wx_i:i_x + self.wx_i,
                       i_y - self.wy_i:i_y + self.wy_i] = True

    @property
    def wx(self):
        return self.wx_i * self.dx

    @property
    def wy(self):
        return self.wy_i * self.dx

    @property
    def sx(self):
        return self.sx_i * self.dx

    @property
    def sy(self):
        return self.sy_i * self.dx

    def __repr__(self):
        fs = [('L', self.L), ('dx', self.dx),
              ('wx', self.wx), ('wy', self.wy),
              ('sx', self.sx), ('sy', self.sy)]
        return make_repr_str(self, fs)


class Traps(Walls):
    """A set of walls forming a number of 2D traps.

    Parameters
    ----------
    n: int
        The number of traps. Can be 1, 4 or 5.
    d: float
        The width of the trap wall.
        Valid values are `i * dx`, where `i` is an integer >= 1.
    w: float
        The width of the entire trap.
        Valid values are `(2i + 1) dx`, where `i` is an integer >= 0.
    s: float
        The width of the trap entrance.
        Valid values are `(2i + 1) dx`, where `i` is an integer >= 0.
    """

    def __init__(self, L, dx, n, d, w, s):
        Walls.__init__(self, L, dim=2, dx=dx)

        # Calculate length in terms of lattice indices
        d_i = int(round(d / self.dx))
        w_i = int(round(w / self.dx))
        s_i = int(round(s / self.dx))

        # Calculate how many indices to go in each direction
        w_i_half = w_i // 2
        s_i_half = s_i // 2
        # l is the width of the trap, including its walls.
        l_i_half = w_i_half + d_i
        # Going to carve out `w_i_half` in each direction from a cell,
        # so will carve out this many cells.
        w_i = 2 * w_i_half + 1
        # Same goes for the slit.
        s_i = 2 * s_i_half + 1

        self.d_i = d_i
        self.w_i = w_i
        self.s_i = s_i

        if self.w < 0.0 or self.w > self.L:
            raise Exception('Invalid trap width')
        if self.s < 0.0 or self.s > self.w:
            raise Exception('Invalid slit length')

        if n == 1:
            traps_f = np.array([[0.50, 0.50]],
                               dtype=np.float)
        elif n == 4:
            traps_f = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25],
                                [0.75, 0.75]],
                               dtype=np.float)
        elif n == 5:
            traps_f = np.array([[0.25, 0.25], [0.25, 0.75],
                                [0.50, 0.50],
                                [0.75, 0.25], [0.75, 0.75]],
                               dtype=np.float)
        else:
            raise Exception('Traps not implemented for this number of traps')

        self.traps_i = np.asarray(self.M * traps_f, dtype=np.int)
        for x, y in self.traps_i:
            # First fill in entire trap-related area.
            self.a[x - l_i_half:x + l_i_half + 1,
                   y - l_i_half:y + l_i_half + 1] = True
            # Then carve out trap interior.
            self.a[x - w_i_half:x + w_i_half + 1,
                   y - w_i_half:y + w_i_half + 1] = False
            # Then make the slit.
            self.a[x - s_i_half:x + s_i_half + 1,
                   y + w_i_half:y + l_i_half + 1] = False

    @property
    def d(self):
        return self.d_i * self.dx

    @property
    def w(self):
        return self.w_i * self.dx

    @property
    def s(self):
        return self.s_i * self.dx

    @property
    def n(self):
        return self.traps_i.shape[0]

    def get_trap_areas_i(self):
        """Calculate the number of elements occupied by each trap.

        Returns
        -------
        trap_areas_i: list[int]
        """
        trap_areas_i = []
        w_i_half = self.w_i // 2
        for x, y in self.traps_i:
            trap = self.a[x - w_i_half:x + w_i_half + 1,
                          y - w_i_half:y + w_i_half + 1]
            trap_areas_i.append(np.sum(np.logical_not(trap)))
        return np.array(trap_areas_i)

    def get_trap_areas(self):
        """Calculate the area occupied by each trap.

        Returns
        -------
        trap_areas: list[float]
        """
        return self.get_trap_areas_i() * self.dA

    def __repr__(self):
        fs = [('L', self.L), ('dx', self.dx),
              ('n', self.n), ('d', self.d), ('w', self.w), ('s', self.s)]
        return make_repr_str(self, fs)


class Maze(Walls):
    """A set of walls forming a maze.

    Parameters
    ----------
    d: float
        The width of the maze walls.
    seed: int
        The random number seed used to generate the maze.
        Note that this does not affect, or is affected by, pre-existing
        random number seeding.
    """

    def __init__(self, L, dim, dx, d, seed=None):
        Walls.__init__(self, L, dim, dx)
        self.seed = seed
        rng = np.random.RandomState(self.seed)
        self.M_m = int(round(self.L / d))
        self.d_i = int(round(self.M / self.M_m))
        a_base = maze.make_maze_dfs(self.M_m, self.dim, rng)
        self.a[...] = lattice.extend_array(a_base, self.d_i)

    @property
    def d(self):
        return self.d_i * self.dx

    def __repr__(self):
        fs = [('L', self.L), ('dim', self.dim), ('dx', self.dx), ('d', self.d),
              ('seed', self.seed)]
        return make_repr_str(self, fs)
