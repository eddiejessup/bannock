from __future__ import print_function, division
import numpy as np
from ciabatta.meta import make_repr_str
from spatious import vector
from cellulist import cell_list
from bannock import numerics, walls
from bannock.secretion import Secretion, WalledSecretion


class AutoBaseModel(object):

    def __init__(self, dim, seed, dt,
                 origin_flag,
                 rho_0, v_0, p_0,
                 chi, onesided_flag,
                 vicsek_R,
                 c_D, c_sink, c_source):
        self.seed = seed
        self.dt = dt
        self.origin_flag = origin_flag
        self.v_0 = v_0
        self.p_0 = p_0
        self.chi = chi
        self.onesided_flag = onesided_flag
        self.vicsek_R = vicsek_R

        np.random.seed(seed)
        self.t = 0.0
        self.i = 0

    def _tumble(self):
        self.p[:] = self.p_0

        if self.chi:
            grad_c_i = self.c.grad_i(self.r)
            v_dot_grad_c = np.sum(self.v * grad_c_i, axis=-1)
            fitness = self.chi * v_dot_grad_c / self.v_0

            self.p *= 1.0 - fitness
            if self.onesided_flag:
                self.p = np.minimum(self.p_0, self.p)
            # self.p = np.maximum(self.p, 0.1)

        tumbles = np.random.uniform(size=self.n) < self.p * self.dt
        self.v[tumbles] = self.v_0 * vector.sphere_pick(self.dim,
                                                        tumbles.sum())

    def iterate(self):
        self.t += self.dt
        self.i += 1

    @property
    def L_half(self):
        return self.L / 2.0

    @property
    def n(self):
        return self.r.shape[0]

    @property
    def dim(self):
        return self.r.shape[1]

    @property
    def c_sink(self):
        return self.c.sink_rate

    @property
    def c_source(self):
        return self.c.source_rate

    @property
    def c_D(self):
        return self.c.D

    @property
    def dx(self):
        return self.c.D


class Model1D(AutoBaseModel):
    """Self-propelled particles moving in one dimension in a chemical field.

    Parameters
    ----------
    seed: int
        A random number seed. `None` causes a random choice.
    dt: float
        The size of a single time-step.
    origin_flag: bool
        Whether or not to start the particles at the centre of the system.
        If `True`, all particles are initialised in a small region near the
        origin.
        If `False`, particles are initialised uniformly.
    rho_0: float
        The average area density of particles.
    v_0: float
        The speed of the particles.
    p_0: float
        The base rate at which the particles randomise their
        direction.
    chi: float
        The sensitivity of the particles' chemotactic response to gradients
        in the chemoattractant concentration field.
    onesided_flag: bool
        Whether or not the particles' chemotactic response can increase
        their tumble rate.
    vicsek_R: float
        A radius within which the particles reorient with their neighbours.
    c_D: float
        see :class:`Secretion`
    c_sink: float
        see :class:`Secretion`
    c_source: float
        see :class:`Secretion`
    L: float
        Length of the system.
    dx: float
        Length of a cell in the chemical concentration field lattice.
    """

    def __init__(self, seed, dt,
                 origin_flag,
                 rho_0, v_0, p_0,
                 chi, onesided_flag,
                 vicsek_R,
                 c_D, c_sink, c_source,
                 L, dx,
                 *args, **kwargs):
        dim = 1
        super(Model1D, self).__init__(dim, seed, dt,
                                      origin_flag,
                                      rho_0, v_0, p_0,
                                      chi, onesided_flag,
                                      vicsek_R,
                                      c_D, c_sink, c_source)
        self.L = L

        n = int(round(self.L * rho_0))

        self.v = self.v_0 * vector.sphere_pick(dim, n)
        if self.origin_flag:
            # Randomise initialisation by a small distance, to avoid
            # unphysical regular spacing otherwise. In 1D the particles are
            # effectively on a lattice of length `v_0 * dt`.
            vdt = self.dt * self.v_0
            self.r = np.random.uniform(-vdt, vdt, [n, dim])
        else:
            self.r = np.random.uniform(-self.L_half, self.L_half, [n, dim])
        self.p = np.ones([self.n]) * self.p_0

        self.c = Secretion(self.L, dim, dx, c_D, self.dt, c_sink,
                           c_source, a_0=0.0)

    @property
    def rho_0(self):
        return self.n / self.L

    def _update_positions(self):
        self.r += self.v * self.dt
        self.r[self.r > self.L_half] -= self.L
        self.r[self.r < -self.L_half] += self.L

    def _vicsek(self):
        u = np.array(np.round(self.v[:, 0] / self.v_0), dtype=np.int)
        u_new = numerics.vicsek_1d(self.r[:, 0], u,
                                   self.vicsek_R, self.L)
        stats = u_new == 0
        u_new[stats] = 2 * np.random.randint(2, size=stats.sum()) - 1
        self.v[:, 0] = self.v_0 * u_new

    def iterate(self):
        """Evolve the model's state by a single time-step.

        - Do Vicsek alignment

        - Make particles tumble at their chemotactic probabilities.

        - Make the particles swim in the periodic space

        - Iterate the chemical concentration field
        """
        if self.vicsek_R:
            self._vicsek()
        if self.p_0:
            self._tumble()
        self._update_positions()

        if self.c_source:
            self.c.iterate(self.r)

        super(Model1D, self).iterate()

    def get_output_dirname(self):
        s = 'Bannock_{}D,seed={},dt={:g},'.format(self.dim, self.seed, self.dt)
        s += 'origin={:d},'.format(self.origin_flag)
        s += 'L={:g},'.format(self.L)
        s += 'rho={:g},v={:g},p={:g},'.format(self.rho_0, self.v_0, self.p_0)
        s += 'chi={:g},side={},'.format(self.chi, 2 - self.onesided_flag)
        s += 'vicsek_R={:g},'.format(self.vicsek_R)
        s += 'c_source={:g},c_sink={:g},c_D,dx={:g}'.format(self.c_source,
                                                            self.c_sink,
                                                            self.c_D,
                                                            self.dx)
        return s

    def __repr__(self):
        fs = [('dim', self.dim), ('seed', self.seed), ('dt', self.dt),
              ('origin_flag', self.origin_flag),
              ('L', self.L),
              ('n', self.n), ('v_0', self.v_0), ('p_0', self.p_0),
              ('chi', self.chi), ('onesided_flag', self.onesided_flag),
              ('vicsek_R', self.vicsek_R),
              ('c', self.c),
              ('t', self.t), ('i', self.i)
              ]
        return make_repr_str(self, fs)


class Model2D(AutoBaseModel):
    """Self-propelled particles moving in two dimensions in a chemical field.

    Parameters
    ----------
    D_rot: float
        The rotational diffusion constant of the particles.
    force_mu: float
        The degree to which the particles reorient towards
        :math:`\\nabla c`, where :math:`c` is the chemoattractant
        concentration field.
    walls: Walls
        Obstacles in the environment
    Others:
        see :class:`Model1D`.
    """

    def __init__(self, seed, dt,
                 origin_flag,
                 rho_0, v_0, p_0,
                 D_rot,
                 chi, onesided_flag,
                 vicsek_R,
                 force_mu,
                 c_D, c_sink, c_source,
                 walls,
                 *args, **kwargs):
        dim = 2
        super(Model2D, self).__init__(dim, seed, dt,
                                      origin_flag,
                                      rho_0, v_0, p_0,
                                      chi, onesided_flag,
                                      vicsek_R,
                                      c_D, c_sink, c_source)
        self.D_rot = D_rot
        self.force_mu = force_mu
        self.walls = walls

        self.c = WalledSecretion(self.walls.L, self.walls.dim, self.walls.dx,
                                 self.walls.a, c_D, self.dt,
                                 c_sink, c_source, a_0=0.0)

        n = int(round(self.walls.free_area * rho_0))

        self.v = self.v_0 * vector.sphere_pick(dim, n)
        self.r = np.zeros_like(self.v)
        if self.origin_flag:
            if self.walls.is_obstructed(self.r[0, np.newaxis]):
                raise Exception('Cannot initialise particles at the origin as '
                                'there is an obstacle there')
        else:
            for i in range(self.n):
                while True:
                    self.r[i] = np.random.uniform(-self.L_half, self.L_half,
                                                  dim)
                    if not self.walls.is_obstructed(self.r[i, np.newaxis]):
                        break
        self.p = np.ones([self.n]) * self.p_0

    @property
    def rho_0(self):
        return self.n / self.walls.free_area

    @property
    def L(self):
        return self.walls.L

    def _update_positions(self):
        r_old = self.r.copy()

        self.r += self.v * self.dt
        self.r[self.r > self.L_half] -= self.L
        self.r[self.r < -self.L_half] += self.L

        if self.walls.a.any():
            obs = self.walls.is_obstructed(self.r)
            # Find particles and dimensions which have changed cell.
            changeds = np.not_equal(self.walls.r_to_i(self.r),
                                    self.walls.r_to_i(r_old))
            # Find where particles have collided with a wall,
            # and the dimensions on which it happened.
            colls = np.logical_and(obs[:, np.newaxis], changeds)

            # Reset particle position components along which a collision
            # occurred
            self.r[colls] = r_old[colls]
            # Set velocity along that axis to zero.
            self.v[colls] = 0.0

            # Rescale new directions, randomising stationary particles.
            self.v[obs] = vector.vector_unit_nullrand(self.v[obs]) * self.v_0

    def _force(self):
        grad_c_i = self.c.grad_i(self.r)
        v_dot_grad_c = np.sum(self.v * grad_c_i, axis=-1)
        if self.onesided_flag:
            responds = v_dot_grad_c > 0.0
        else:
            responds = np.ones([self.n], dtype=np.bool)
        self.v[responds] += self.force_mu * grad_c_i[responds] * self.dt
        self.v = self.v_0 * vector.vector_unit_nullnull(self.v)

    def _rot_diff(self):
        self.v = numerics.rot_diff_2d(self.v, self.D_rot, self.dt)

    def _vicsek(self):
        inters, intersi = cell_list.get_inters(self.r, self.L, self.vicsek_R)
        self.v = numerics.vicsek_inters(self.v, inters, intersi)

    def iterate(self):
        """Evolve the model's state by a single time-step.

        - Do Vicsek alignment

        - Make particles tumble at their chemotactic probabilities.

        - Reorient particles according to chemotactic gradient

        - Diffuse the particles' directions

        - Make the particles swim in the periodic space

        - Reorient the particles that collide with walls, and move them back
          to their original positions

        - Iterate the chemical concentration field
        """
        if self.vicsek_R:
            self._vicsek()
        if self.p_0:
            self._tumble()
        if self.force_mu:
            self._force()
        if self.D_rot:
            self._rot_diff()
        self._update_positions()

        if self.c_source:
            self.c.iterate(self.r)
        super(Model2D, self).iterate()

    def get_output_dirname_walls_part(self):
        w = self.walls
        if w.__class__ == walls.Walls:
            return 'Walls'
        elif w.__class__ == walls.Traps:
            return 'Traps_n={},d={:g},w={:g},s={:g}'.format(w.n, w.d, w.w, w.s)
        elif w.__class__ == walls.Maze:
            return 'Maze_d={:g},seed={}'.format(w.d, w.seed)

    def get_output_dirname(self):
        s = 'Bannock_{}D,seed={},dt={:g},'.format(self.dim, self.seed, self.dt)
        s += 'origin={:d},'.format(self.origin_flag)
        s += 'walls={},'.format(self.get_output_dirname_walls_part())
        s += 'rho={:g},v={:g},p={:g},'.format(self.rho_0, self.v_0, self.p_0)
        s += 'Dr={:g},'.format(self.D_rot)
        s += 'chi={:g},side={},'.format(self.chi, 2 - self.onesided_flag)
        s += 'vicsek_R={:g},'.format(self.vicsek_R)
        s += 'force_mu={:g},'.format(self.force_mu)
        s += 'c_source={:g},c_sink={:g},c_D,dx={:g}'.format(self.c_source,
                                                            self.c_sink,
                                                            self.c_D,
                                                            self.dx)
        return s

    def __repr__(self):
        fs = [('dim', self.dim), ('seed', self.seed), ('dt', self.dt),
              ('origin_flag', self.origin_flag),
              ('walls', self.walls),
              ('n', self.n), ('v_0', self.v_0), ('p_0', self.p_0),
              ('D_rot', self.D_rot),
              ('chi', self.chi), ('onesided_flag', self.onesided_flag),
              ('vicsek_R', self.vicsek_R),
              ('force_mu', self.force_mu),
              ('c', self.c),
              ('t', self.t), ('i', self.i)
              ]
        return make_repr_str(self, fs)


class Model2DNoAlignment(Model2D):
    """Self-propelled particles moving in two dimensions in a chemical field.

    Particles do not align with obstacles, but instead reflect off.

    Parameters
    ----------
    see :class:`Model2D`.
    """

    def _update_positions(self):
        r_old = self.r.copy()

        self.r += self.v * self.dt
        self.r[self.r > self.L_half] -= self.L
        self.r[self.r < -self.L_half] += self.L

        if self.walls.a.any():
            obs = self.walls.is_obstructed(self.r)
            # Find particles and dimensions which have changed cell.
            changeds = np.not_equal(self.walls.r_to_i(self.r),
                                    self.walls.r_to_i(r_old))
            # Find where particles have collided with a wall,
            # and the dimensions on which it happened.
            colls = np.logical_and(obs[:, np.newaxis], changeds)

            # Reset particle position components along which a collision
            # occurred
            self.r[colls] = r_old[colls]
            # Reflect velocity along that axis.
            self.v[colls] = -self.v[colls]

            # Rescale new directions, randomising stationary particles.
            self.v[obs] = vector.vector_unit_nullrand(self.v[obs]) * self.v_0
