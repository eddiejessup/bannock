from abc import ABCMeta, abstractmethod
import numpy as np
from ciabatta import vector, fields
from ciabatta.cell_list import intro
import particle_numerics
from utils import reprify
from secretion import Secretion, WalledSecretion


class BaseModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, dim, seed):
        self.dim = dim
        self.seed = seed

        self.t = 0.0
        self.i = 0

        np.random.seed(self.seed)

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

    def get_density_field(self):
        """Calculate a field on the same lattice as the chemical concentration,
        for the particle number density, binning the particles into their
        nearest cell.

        Returns
        -------
        d: numpy.ndarray[dtype=int]
            Density field
        """
        return fields.density(self.r, self.L, self.c.dx())


class Model1D(BaseModel):
    """Self-propelled particles moving in one dimension in a chemical field.

    Parameters
    ----------
    seed: int
        A random number seed. `None` causes a random choice.
    dt: float
        The size of a single time-step.
    rho_0: float
        The average area density of particles.
    v_0: float
        The speed of the particles.
    p_0: float
        The base rate at which the particles randomise their
        direction.
    origin_flag: bool
        Whether or not to start the particles at the centre of the system.
        If `True`, all particles are initialised in a small region near the
        origin.
        If `False`, particles are initialised uniformly.
    chi: float
        The sensitivity of the particles' chemotactic response to gradients
        in the chemoattractant concentration field.
    onesided_flag: bool
        Whether or not the particles' chemotactic response can increase
        their tumble rate.
    vicsek_R: float
        A radius within which the particles reorient with their neighbours.
    L: float
        Length of the system.
    dx: float
        Length of a cell in the chemical concentration field lattice.
    c_D: float
        see :class:`Secretion`
    c_sink: float
        see :class:`Secretion`
    c_source: float
        see :class:`Secretion`
    """
    def __init__(self, seed, dt,
                 rho_0, v_0, p_0, origin_flag,
                 chi, onesided_flag,
                 vicsek_R,
                 L, dx,
                 c_D, c_sink, c_source,
                 *args, **kwargs):
        BaseModel.__init__(self, 1, seed)
        self.dt = dt
        self.v_0 = v_0
        self.p_0 = p_0
        self.origin_flag = origin_flag
        self.chi = chi
        self.onesided_flag = onesided_flag
        self.vicsek_R = vicsek_R
        self.L = L
        self.L_half = self.L / 2.0
        self.dx = dx
        self.c_D = c_D
        self.c_sink = c_sink
        self.c_source = c_source

        if self.c_source:
            self.c = Secretion(self.L, self.dim, self.dx,
                               self.c_D, self.dt,
                               self.c_sink, self.c_source, a_0=0.0)

        self.n = int(round(self.L * rho_0))
        self.rho_0 = self.n / self.L

        self.v = self.v_0 * vector.sphere_pick(self.dim, self.n)
        self._initialise_r()
        self.p = np.ones([self.n]) * self.p_0

    def _initialise_r(self):
        if self.origin_flag:
            # Randomise initialisation by a small distance, to avoid
            # unphysical regular spacing otherwise. In 1D the particles are
            # effectively on a lattice of length `v_0 * dt`.
            vdt = self.dt * self.v_0
            self.r = np.random.uniform(-vdt, vdt, [self.n, self.dim])
        else:
            self.r = np.random.uniform(-self.L_half, self.L_half,
                                       [self.n, self.dim])

    def _update_positions(self):
        self.r += self.v * self.dt
        self.r[self.r > self.L_half] -= self.L
        self.r[self.r < -self.L_half] += self.L

    def _vicsek(self):
        u = np.array(np.round(self.v[:, 0] / self.v_0), dtype=np.int)
        u_new = particle_numerics.vicsek_1d(self.r[:, 0], u,
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
            density = self.get_density_field()
            self.c.iterate(density)

        self.t += self.dt
        self.i += 1

    def __repr__(self):
        fields = ['dim', 'seed', 'dt', 'L', 'dx',
                  'c_D', 'c_sink', 'c_source',
                  'v_0', 'p_0', 'origin_flag',
                  'rho_0',
                  'chi', 'onesided_flag',
                  'vicsek_R',
                  ]
        field_strs = reprify(self, fields)
        return 'autochemo_model_{}'.format(','.join(field_strs))


class Model2D(BaseModel):
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
                 rho_0, v_0, D_rot, p_0, origin_flag,
                 chi, onesided_flag,
                 force_mu,
                 vicsek_R,
                 walls,
                 c_D, c_sink, c_source,
                 *args, **kwargs):
        BaseModel.__init__(self, 2, seed)
        self.dt = dt
        self.v_0 = v_0
        self.D_rot = D_rot
        self.p_0 = p_0
        self.origin_flag = origin_flag
        self.chi = chi
        self.onesided_flag = onesided_flag
        self.force_mu = force_mu
        self.vicsek_R = vicsek_R
        self.walls = walls
        self.L = walls.L
        self.L_half = self.L / 2.0
        self.dx = walls.dx()
        self.c_D = c_D
        self.c_sink = c_sink
        self.c_source = c_source

        self.c = WalledSecretion(self.walls.L, self.walls.dim, self.walls.dx(),
                                 self.walls.a, self.c_D, self.dt,
                                 self.c_sink, self.c_source, a_0=0.0)

        self.n = int(round(self.walls.get_free_area() * rho_0))
        self.rho_0 = self.n / self.walls.get_free_area()

        self.v = self.v_0 * vector.sphere_pick(self.dim, self.n)
        self._initialise_r()
        self.p = np.ones([self.n]) * self.p_0

    def _initialise_r(self):
        self.r = np.zeros_like(self.v)
        if self.origin_flag:
            if self.walls.is_obstructed(self.r[0, np.newaxis]):
                raise Exception('Cannot initialise particles at the origin as'
                                'there is an obstacle there')
        else:
            for i in range(self.n):
                while True:
                    self.r[i] = np.random.uniform(-self.L_half, self.L_half,
                                                  self.dim)
                    if not self.walls.is_obstructed(self.r[i, np.newaxis]):
                        break

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
        self.v = particle_numerics.rot_diff_2d(self.v, self.D_rot, self.dt)

    def _vicsek(self):
        inters, intersi = intro.get_inters(self.r, self.L, self.vicsek_R)
        self.v = particle_numerics.vicsek_inters(self.v, inters, intersi)

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
            density = self.get_density_field()
            self.c.iterate(density)

        self.t += self.dt
        self.i += 1

    def __repr__(self):
        fields = ['dim', 'seed', 'dt', 'L', 'dx',
                  'c_D', 'c_sink', 'c_source',
                  'v_0', 'p_0', 'D_rot', 'origin_flag',
                  'rho_0',
                  'chi', 'onesided_flag',
                  'force_mu', 'vicsek_R',
                  ]
        field_strs = reprify(self, fields)
        field_strs.append(repr(self.walls))
        return 'autochemo_model_{}'.format(','.join(field_strs))
