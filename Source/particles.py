import numpy as np
from ciabatta import utils, fields
from ciabatta.cell_list import intro
import particle_numerics


def get_K(t, dt, tau):
    A = 0.5
    t_s = np.arange(0.0, t, dt)
    g_s = t_s / tau
    K = np.exp(-g_s) * (1.0 - A * (g_s + (g_s ** 2) / 2.0))
    K[K < 0.0] *= np.abs(K[K >= 0.0].sum() / K[K < 0.0].sum())
    K /= np.sum(K * -t_s * dt)
    return K


class Particles(object):
    def __init__(self, L, dim, dt, n, v_0,
                 walls,
                 p_0, chi, dt_chemo, onesided_flag, memory_flag, t_mem,
                 force_chi,
                 D_rot,
                 vicsek_R):
        self.walls = walls
        self.L = L
        self.L_half = self.L / 2.0
        self.dim = dim
        self.dt = dt
        self.n = n
        self.v_0 = v_0
        self.p_0 = p_0
        self.chi = chi
        self.dt_chemo = dt_chemo
        self.onesided_flag = onesided_flag
        self.memory_flag = memory_flag
        self.t_mem = t_mem
        self.force_chi = force_chi
        self.D_rot = D_rot
        self.vicsek_R = vicsek_R

        # Calculate best dt_chemo that can be managed
        # given that it must be an integer multiple of dt.
        # Update chemotaxis every so many iterations
        self.every_chemo = int(round(self.dt_chemo / self.dt))
        # derive effective dt_chemo from this
        self.dt_chemo = self.every_chemo * self.dt

        if self.memory_flag:
            tau_0 = 1.0 / self.p_0
            self.K_dt_chemo = get_K(self.t_mem, self.dt_chemo,
                                    tau_0) * self.dt_chemo
            self.c_mem = np.zeros([self.n, len(self.K_dt_chemo)])

        self.v = self.v_0 * utils.sphere_pick(self.dim, self.n)
        self.initialise_r()

    def initialise_r(self):
        self.r = np.zeros_like(self.v)
        for i in range(self.n):
            while True:
                self.r[i] = np.random.uniform(-self.L_half, self.L_half,
                                              self.dim)
                if not self.walls.is_obstructed(self.r[i]):
                    break

    def iterate(self, c):
        if self.vicsek_R:
            self.vicsek()
        if self.p_0:
            self.tumble(c)
        if self.force_chi:
            self.force(c)
        if self.D_rot:
            self.rot_diff()
        self.update_positions()

    def update_positions(self):
        r_old = self.r.copy()

        self.r += self.v * self.dt
        self.r[self.r > self.L_half] -= self.L
        self.r[self.r < -self.L_half] += self.L

        obstructeds = self.walls.is_obstructed(self.r)
        # Find particles and dimensions which have changed cell.
        changeds = np.not_equal(self.walls.r_to_i(self.r),
                                self.walls.r_to_i(r_old))
        # Find where particles have collided with a wall,
        # and the dimensions on which it happened.
        collideds = np.logical_and(obstructeds[:, np.newaxis], changeds)

        # Reset particle position components along which a collision occurred
        self.r[collideds] = r_old[collideds]
        # Set velocity along that axis to zero.
        self.v[collideds] = 0.0

        # Make sure no particles are left obstructed.
        assert not self.walls.is_obstructed(self.r).any()
        # Rescale new directions, randomising stationary particles.
        self.v = utils.vector_unit_nullrand(self.v) * self.v_0

    def update_tumble_rate(self, c):
        if self.memory_flag:
            self.c_mem[:, 1:] = self.c_mem.copy()[:, :-1]
            self.c_mem[:, 0] = utils.field_subset(c.a, c.r_to_i(self.r))
            v_dot_grad_c = np.sum(self.c_mem * self.K_dt_chemo, axis=1)
        else:
            grad_c_i = c.grad_i(self.r)
            v_dot_grad_c = np.sum(self.v * grad_c_i, axis=-1)

        fitness = self.chi * v_dot_grad_c / self.v_0

        self.p = self.p_0 * (1.0 - fitness)
        if self.onesided_flag:
            self.p = np.minimum(self.p_0, self.p)

    def tumble(self, c):
        # Update tumble rate every `every_chemo` iterations
        if not self.i % self.every_chemo:
            self.update_tumble_rate(c)
        tumbles = np.random.uniform(size=self.n) < self.p * self.dt
        self.v[tumbles] = self.v_0 * utils.sphere_pick(self.dim, tumbles.sum())

    def force(self, c):
        grad_c_i = c.grad_i(self.r)
        v_dot_grad_c = np.sum(self.v * grad_c_i, axis=-1)
        going_up = v_dot_grad_c > 0.0
        self.v[going_up] += self.force_chi * grad_c_i[going_up] * self.dt
        self.v = self.v_0 * utils.vector_unit_nullnull(self.v)

    def rot_diff(self):
        self.v = utils.rot_diff(self.v, self.D_rot, self.dt)

    def vicsek(self):
        inters, intersi = intro.get_inters(self.r, self.L, self.vicsek_R)
        self.v = particle_numerics.vicsek_inters(self.v, inters, intersi)

    def get_density_field(self, dx):
        return fields.density(self.r, self.L, dx)
