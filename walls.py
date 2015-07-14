from __future__ import print_function, division
import numpy as np
from ciabatta import lattice, fields, maze


class Walls(fields.Field):
    """A field representing an environment filled with square obstacles."""

    def __init__(self, L, dim, dx):
        fields.Field.__init__(self, L, dim, dx)
        self.a = np.zeros(self.dim * (self.M,), dtype=np.uint8)
        self.d = self.L_half

    def get_free_area_i(self):
        """Calculate the number of elements that are not occupied by obstacles.

        Returns
        -------
        area_i: int
        """
        return np.logical_not(self.a).sum()

    def get_free_area(self):
        """Calculate the area that is not occupied by obstacles.

        Returns
        -------
        area: float
        """
        return self.get_free_area_i() * self.dA()

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


class HalfClosed(Walls):
    """A set of walls closing the 2D environment along one edge."""

    def __init__(self, L, dx):
        Walls.__init__(self, L, dim=2, dx=dx)
        self.a[:, 0] = True


class Closed(Walls):
    """A set of walls closing the 2D environment at all edges."""

    def __init__(self, L, dx):
        Walls.__init__(self, L, dim=2, dx=dx)
        self.a[:, 0] = True
        self.a[:, -1] = True
        self.a[0, :] = True
        self.a[-1, :] = True


class Tittled(Walls):
    repr_fields = Walls.repr_fields + ['wx', 'wy', 'sx', 'sy']

    def __init__(self, L, dx, wx, wy, sx, sy):
        Walls.__init__(self, L, dim=2, dx=dx)
        self.wx_i = int(round(wx / self.dx))
        self.wy_i = int(round(wy / self.dx))
        self.sx_i = int(round(sx / self.dx))
        self.sy_i = int(round(sy / self.dx))
        self.wx = self.wx_i * self.dx
        self.wy = self.wy_i * self.dx
        self.sx = self.sx_i * self.dx
        self.sy = self.sy_i * self.dx

        for i_x in range(self.sx_i,
                         self.a.shape[0] - self.sx_i,
                         self.sx_i):
            for i_y in range(self.sy_i,
                             self.a.shape[1] - self.sy_i,
                             self.sy_i):
                self.a[i_x - self.wx_i:i_x + self.wx_i,
                       i_y - self.wy_i:i_y + self.wy_i] = True


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
    repr_fields = Walls.repr_fields + ['n', 'd', 'w', 's']

    def __init__(self, L, dx, n, d, w, s):
        Walls.__init__(self, L, dim=2, dx=dx)
        self.n = n

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

        # Scale back up to physical lengths.
        self.d = self.d_i * self.dx
        self.w = self.w_i * self.dx
        self.s = self.s_i * self.dx

        if self.w < 0.0 or self.w > self.L:
            raise Exception('Invalid trap width')
        if self.s < 0.0 or self.s > self.w:
            raise Exception('Invalid slit length')

        if self.n == 1:
            self.traps_f = np.array([[0.50, 0.50]], dtype=np.float)
        elif self.n == 4:
            self.traps_f = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25],
                                    [0.75, 0.75]], dtype=np.float)
        elif self.n == 5:
            self.traps_f = np.array([[0.25, 0.25], [0.25, 0.75],
                                     [0.50, 0.50],
                                     [0.75, 0.25], [0.75, 0.75]],
                                    dtype=np.float)
        else:
            raise Exception('Traps not implemented for this number of traps')

        self.traps_i = np.asarray(self.M * self.traps_f, dtype=np.int)
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
        return self.get_trap_areas_i() * self.dA()

    def get_fracs(self, r):
        """Calculate the fraction of particles inside each trap.

        Parameters
        ----------
        r: array_like[shape=(n, 2)]
            Particle position vectors.

        Returns
        -------
        fracs: list[int]
            Fraction of the total population that is inside each trap.
        """
        inds = self.r_to_i(r)
        n_traps = [0 for i in range(len(self.traps_i))]
        w_i_half = self.w_i // 2
        for i_trap in range(len(self.traps_i)):
            mid_x, mid_y = self.traps_i[i_trap]

            low_x, high_x = mid_x - w_i_half, mid_x + w_i_half
            low_y, high_y = mid_y - w_i_half, mid_y + w_i_half
            for i_x, i_y in inds:
                if low_x <= i_x <= high_x and low_y <= i_y <= high_y:
                    n_traps[i_trap] += 1
        return np.array([float(n_trap) / float(r.shape[0])
                         for n_trap in n_traps])


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
    repr_fields = Walls.repr_fields + ['d', 'seed']

    def __init__(self, L, dim, dx, d, seed=None):
        Walls.__init__(self, L, dim, dx)
        self.seed = seed
        self.M_m = int(round(self.L / d))
        self.d_i = int(round(self.M / self.M_m))
        self.d = self.d_i * self.dx
        a_base = maze.make_maze_dfs(self.M_m, self.dim, self.seed)
        self.a[...] = lattice.extend_array(a_base, self.d_i)
