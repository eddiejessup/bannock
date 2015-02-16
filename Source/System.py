import numpy as np
import particles
from ciabatta import fields


class Secretion(fields.WalledDiffusing):
    def __init__(self, L, dim, dx, walls, D, dt, sink_rate, source_rate,
                 a_0=0.0):
        fields.WalledDiffusing.__init__(self, L, dim, dx, walls, D, dt,
                                        a_0=a_0)
        self.source_rate = source_rate
        self.sink_rate = sink_rate

    def iterate(self, density, food):
        fields.WalledDiffusing.iterate(self)
        self.a += self.dt * (self.source_rate * density * food -
                             self.sink_rate * self.a)
        self.a = np.maximum(self.a, 0.0)


class System(object):
    def __init__(self, seed, dt,
                 walls,
                 rho_0, v_0,
                 f_0, c_D, c_sink, c_source,
                 p_0, memory_flag, chi, t_mem, onesided_flag,
                 force_chi,
                 D_rot,
                 vicsek_R):
        self.seed = seed
        self.dt = dt
        self.t = 0.0
        self.i = 0

        np.random.seed(self.seed)

        self.f = fields.WalledScalar(walls.L, walls.dim,
                                     walls.dx(), walls.a, f_0)
        self.c = Secretion(walls.L, walls.dim, walls.dx(),
                           walls.a, c_D, self.dt, c_sink, c_source,
                           a_0=0.0)

        n = int(round(walls.get_A_free() * rho_0))

        self.particles = particles.Particles(walls.L, walls.dim,
                                             self.dt, n, v_0,
                                             walls,
                                             p_0, memory_flag, chi,
                                             t_mem, onesided_flag,
                                             force_chi,
                                             D_rot,
                                             vicsek_R)

    def iterate(self):
        self.particles.iterate(self.c)
        density = self.particles.get_density_field(self.f.dx())
        self.c.iterate(density, self.f.a)
        self.t += self.dt
        self.i += 1
