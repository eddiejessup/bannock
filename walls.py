from collections import OrderedDict
import numpy as np
from ciabatta import lattice, fields, maze
from model import format_parameter

BUFFER_SIZE = 0.999


class Walls(fields.Field):
    def __init__(self, L, dim, dx):
        fields.Field.__init__(self, L, dim, dx)
        self.a = np.zeros(self.dim * (self.M,), dtype=np.uint8)
        self.d = self.L_half

    def get_A_free_i(self):
        return np.logical_not(self.a).sum()

    def get_A_free(self):
        return self.A() * float(self.get_A_free_i()) / self.A_i()

    def is_obstructed(self, r):
        return self.a[tuple(self.r_to_i(r).T)]

    def __repr__(self):
        return 'walls_blank'


class HalfClosed(Walls):
    def __init__(self, L, dim, dx):
        Walls.__init__(self, L, dim, dx)
        self.a[:, 0] = True
        self.a[:, -1] = True


class Closed(Walls):
    def __init__(self, L, dim, dx):
        Walls.__init__(self, L, dim, dx)
        self.a[:, 0] = True
        self.a[:, -1] = True
        self.a[0, :] = True
        self.a[-1, :] = True


class Tittled(Walls):
    def __init__(self, L, dim, dx, wx, wy, sx, sy):
        Walls.__init__(self, L, dim, dx)
        self.wx_i = int(round(wx / self.dx()))
        self.wy_i = int(round(wy / self.dx()))
        self.sx_i = int(round(sx / self.dx()))
        self.sy_i = int(round(sy / self.dx()))
        self.wx = self.wx_i * self.dx()
        self.wy = self.wy_i * self.dx()
        self.sx = self.sx_i * self.dx()
        self.sy = self.sy_i * self.dx()

        for i_x in range(self.sx_i,
                         self.a.shape[0] - self.sx_i,
                         self.sx_i):
            print(i_x)
            for i_y in range(self.sy_i,
                             self.a.shape[1] - self.sy_i,
                             self.sy_i):
                print(i_y)
                self.a[i_x - self.wx_i:i_x + self.wx_i,
                       i_y - self.wy_i:i_y + self.wy_i] = True
        print(self.a.max())


class Traps(Walls):
    def __init__(self, L, dim, dx, n, d, w, s):
        '''Make a set of walls making a number of traps.

        d: Wall width.
        w: Trap width.
        s: Trap entrance width.
        '''

        Walls.__init__(self, L, dim, dx)
        self.n = n
        self.d_i = int(d / self.dx()) + 1
        self.w_i = int(w / self.dx()) + 1
        self.s_i = int(s / self.dx()) + 1
        self.d = self.d_i * self.dx()
        self.w = self.w_i * self.dx()
        self.s = self.s_i * self.dx()

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

        w_i_half = self.w_i // 2
        s_i_half = self.s_i // 2
        self.traps_i = np.asarray(self.M * self.traps_f, dtype=np.int)
        for x, y in self.traps_i:
            self.a[x - w_i_half - self.d_i:x + w_i_half + self.d_i + 1,
                   y - w_i_half - self.d_i:y + w_i_half + self.d_i + 1] = True
            self.a[x - w_i_half:x + w_i_half + 1,
                   y - w_i_half:y + w_i_half + 1] = False
            self.a[x - s_i_half:x + s_i_half + 1,
                   y + w_i_half:y + w_i_half + self.d_i + 1] = False

    def get_A_traps_i(self):
        A_traps_i = 0
        w_i_half = self.w_i // 2
        for x, y in self.traps_i:
            trap = self.a[x - w_i_half:x + w_i_half + 1,
                          y - w_i_half:y + w_i_half + 1]
            A_traps_i += np.sum(np.logical_not(trap))
        return A_traps_i

    def get_A_traps(self):
        return self.A() * float(self.get_A_traps_i()) / self.get_A_free_i()

    def get_fracs(self, r):
        inds = self.r_to_i(r)
        n_traps = [0 for i in range(len(self.traps_i))]
        w_i_half = self.w_i // 2
        for i_trap in range(len(self.traps_i)):
            mid_x, mid_y = self.traps_i[i_trap]

            low_x, high_x = mid_x - w_i_half, mid_x + w_i_half
            low_y, high_y = mid_y - w_i_half, mid_y + w_i_half
            for i_x, i_y in inds:
                if low_x < i_x < high_x and low_y < i_y < high_y:
                    n_traps[i_trap] += 1
        return [float(n_trap) / float(r.shape[0]) for n_trap in n_traps]

    def __repr__(self):
        fields = ['n', 'd', 'w', 's']
        field_vals = OrderedDict([(f, format_parameter(self.__dict__[f]))
                                  for f in fields])
        field_strs = ['='.join([f, v]) for f, v in field_vals.items()]
        field_str = ','.join(field_strs)
        return 'walls_traps_{}'.format(field_str)


class Maze(Walls):
    def __init__(self, L, dim, dx, d, seed=None):
        Walls.__init__(self, L, dim, dx)
        if self.L / self.dx() % 1 != 0:
            raise Exception('Require L / dx to be an integer')
        if self.L / self.d % 1 != 0:
            raise Exception('Require L / d to be an integer')
        if (self.L / self.dx()) / (self.L / self.d) % 1 != 0:
            raise Exception('Require array size / maze size to be integer')

        self.seed = seed
        self.d = d

        self.M_m = int(self.L / self.d)
        self.d_i = int(self.M / self.M_m)
        a_base = maze.make_maze_dfs(self.M_m, self.dim, self.seed)
        self.a[...] = lattice.extend_array(a_base, self.d_i)

    def __repr__(self):
        fields = ['d', 'seed']
        field_vals = OrderedDict([(f, format_parameter(self.__dict__[f]))
                                  for f in fields])
        field_strs = ['='.join([f, v]) for f, v in field_vals.items()]
        field_str = ','.join(field_strs)
        return 'walls_maze_{}'.format(field_str)
