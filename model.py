import numpy as np
from ciabatta import vector, fields
from ciabatta.cell_list import intro
import particle_numerics


class Secretion(fields.Diffusing):
    def __init__(self, L, dim, dx, D, dt, sink_rate, source_rate,
                 a_0=0.0):
        fields.Diffusing.__init__(self, L, dim, dx, D, dt,
                                  a_0=a_0)
        self.source_rate = source_rate
        self.sink_rate = sink_rate

    def iterate(self, density):
        fields.Diffusing.iterate(self)
        self.a += self.dt * (self.source_rate * density -
                             self.sink_rate * self.a)
        self.a = np.maximum(self.a, 0.0)


class WalledSecretion(fields.WalledDiffusing):
    def __init__(self, L, dim, dx, walls, D, dt, sink_rate, source_rate,
                 a_0=0.0):
        fields.WalledDiffusing.__init__(self, L, dim, dx, walls, D, dt,
                                        a_0=a_0)
        self.source_rate = source_rate
        self.sink_rate = sink_rate

    def iterate(self, density):
        fields.WalledDiffusing.iterate(self)
        self.a += self.dt * (self.source_rate * density -
                             self.sink_rate * self.a)
        self.a = np.maximum(self.a, 0.0)


def format_parameter(p):
    if isinstance(p, float):
        return '{:.3g}'.format(p)
    elif p is None:
        return 'N'
    elif isinstance(p, bool):
        return '{:d}'.format(p)
    else:
        return '{}'.format(p)


class Model(object):
    def __init__(self, seed, dt,
                 rho_0, v_0, D_rot, p_0,
                 chi, onesided_flag,
                 force_mu,
                 vicsek_R,
                 walls,
                 c_D, c_sink, c_source):
        self.seed = seed
        self.dt = dt
        self.walls = walls
        self.L = walls.L
        self.L_half = self.L / 2.0
        self.dim = walls.dim
        self.dt = dt
        self.v_0 = v_0
        self.D_rot = D_rot
        self.p_0 = p_0
        self.chi = chi
        self.onesided_flag = onesided_flag
        self.force_mu = force_mu
        self.vicsek_R = vicsek_R
        self.c_D = c_D
        self.c_sink = c_sink
        self.c_source = c_source

        self.t = 0.0
        self.i = 0

        np.random.seed(self.seed)

        self.c = WalledSecretion(self.walls.L, self.walls.dim, self.walls.dx(),
                                 self.walls.a, self.c_D, self.dt,
                                 self.c_sink, self.c_source, a_0=0.0)

        self.n = int(round(self.walls.get_A_free() * rho_0))
        self.rho_0 = self.n / self.walls.get_A_free()

        self.v = self.v_0 * vector.sphere_pick(self.dim, self.n)
        self.initialise_r()
        self.p = np.ones([self.n]) * self.p_0

    def initialise_r(self):
        self.r = np.zeros_like(self.v)
        for i in range(self.n):
            while True:
                self.r[i] = np.random.uniform(-self.L_half, self.L_half,
                                              self.dim)
                if not self.walls.is_obstructed(self.r[i]):
                    break

    def update_positions(self):
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

    def tumble(self):
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

    def force(self):
        grad_c_i = self.c.grad_i(self.r)
        v_dot_grad_c = np.sum(self.v * grad_c_i, axis=-1)
        if self.onesided_flag:
            responds = v_dot_grad_c > 0.0
        else:
            responds = np.ones([self.n], dtype=np.bool)
        self.v[responds] += self.force_mu * grad_c_i[responds] * self.dt
        self.v = self.v_0 * vector.vector_unit_nullnull(self.v)

    def rot_diff(self):
        self.v = particle_numerics.rot_diff_2d(self.v, self.D_rot, self.dt)

    def vicsek(self):
        inters, intersi = intro.get_inters(self.r, self.L, self.vicsek_R)
        self.v = particle_numerics.vicsek_inters(self.v, inters, intersi)

    def iterate(self):
        if self.vicsek_R:
            self.vicsek()
        if self.p_0:
            self.tumble()
        if self.force_mu:
            self.force()
        if self.D_rot:
            self.rot_diff()
        self.update_positions()

        if self.c_source:
            density = self.get_density_field()
            self.c.iterate(density)

        self.t += self.dt
        self.i += 1

    def get_density_field(self):
        return fields.density(self.r, self.L, self.c.dx())


class Model1D(object):
    def __init__(self, seed, dt,
                 rho_0, v_0, p_0,
                 chi, onesided_flag,
                 vicsek_R,
                 L, dx,
                 c_D, c_sink, c_source):
        self.seed = seed
        self.dt = dt
        self.dim = 1
        self.v_0 = v_0
        self.p_0 = p_0

        self.chi = chi
        self.onesided_flag = onesided_flag

        self.vicsek_R = vicsek_R

        self.L = L
        self.L_half = self.L / 2.0
        self.dx = dx

        self.c_D = c_D
        self.c_sink = c_sink
        self.c_source = c_source

        self.t = 0.0
        self.i = 0

        np.random.seed(self.seed)

        if self.c_source:
            self.c = Secretion(self.L, self.dim, self.dx,
                               self.c_D, self.dt,
                               self.c_sink, self.c_source, a_0=0.0)

        self.n = int(round(self.L * rho_0))
        self.rho_0 = self.n / self.L

        self.v = self.v_0 * vector.sphere_pick(self.dim, self.n)
        self.initialise_r()
        self.p = np.ones([self.n]) * self.p_0

    def initialise_r(self):
        self.r = np.random.uniform(-self.L_half, self.L_half,
                                   [self.n, self.dim])

    def update_positions(self):
        self.r += self.v * self.dt
        self.r[self.r > self.L_half] -= self.L
        self.r[self.r < -self.L_half] += self.L

    def tumble(self):
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

    def vicsek(self):
        u = np.array(np.round(self.v[:, 0] / self.v_0), dtype=np.int)
        u_new = particle_numerics.vicsek_1d(self.r[:, 0], u,
                                            self.vicsek_R, self.L)
        stats = u_new == 0
        u_new[stats] = 2 * np.random.randint(2, size=stats.sum()) - 1
        self.v[:, 0] = self.v_0 * u_new

    def iterate(self):
        if self.vicsek_R:
            self.vicsek()
        if self.p_0:
            self.tumble()
        self.update_positions()

        if self.c_source:
            density = self.get_density_field()
            self.c.iterate(density)

        self.t += self.dt
        self.i += 1

    def get_density_field(self):
        return fields.density(self.r, self.L, self.c.dx())


class RampModel1D(Model1D):
    def __init__(self, ramp_chi_0, ramp_chi_max, ramp_dchi_dt, ramp_t_steady,
                 ramp_dt,
                 *args, **kwargs):
        Model1D.__init__(self, *args, **kwargs)

        self.ramp_chi_0 = ramp_chi_0
        self.ramp_chi_max = ramp_chi_max
        self.ramp_dchi_dt = ramp_dchi_dt
        self.ramp_t_steady = ramp_t_steady
        self.ramp_dt = ramp_dt
        self.ramp_chi_func = make_ramp_chi_func(ramp_chi_0, ramp_chi_max,
                                                ramp_dchi_dt, ramp_t_steady,
                                                ramp_dt)
        self.chi = self.ramp_chi_func(self.t)[0]

    def iterate(self):
        Model1D.iterate(self)
        self.chi = self.ramp_chi_func(self.t)[0]

    def __getstate__(self):
        state_dict = self.__dict__.copy()
        del state_dict['ramp_chi_func']
        return state_dict

    def __setstate__(self, state_dict):
        self.__dict__ = state_dict
        self.ramp_chi_func = make_ramp_chi_func(self.ramp_chi_0,
                                                self.ramp_chi_max,
                                                self.ramp_dchi_dt,
                                                self.ramp_t_steady,
                                                self.ramp_dt)


def make_ramp_chi_func(chi_0, chi_max, dchi_dt, t_steady, dt):
    ramp_t_switch = (chi_max - chi_0) / dchi_dt

    def ramp_chi(t):
        ramp_t_raw = t - t_steady

        if ramp_t_raw < ramp_t_switch:
            ramp_t_raw_wrap = ramp_t_raw
        else:
            ramp_t_raw_wrap = 2.0 * ramp_t_switch - ramp_t_raw

        ramp_t_wrap = round(ramp_t_raw_wrap / dt) * dt

        if t < t_steady:
            chi = chi_0
        else:
            chi = chi_0 + ramp_t_wrap * dchi_dt
        return chi, ramp_t_raw_wrap
    return ramp_chi
