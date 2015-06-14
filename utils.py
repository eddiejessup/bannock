from __future__ import print_function, division
import cProfile
import numpy as np
import model
import runner
from runner import filename_to_model, get_filenames
import walls


def dstd_func(d):
    return np.sqrt(np.sum(np.square(d - 1.0)) / d.shape[0])


def density_norm(m):
    d = m.get_density_field() / float(m.rho_0)
    try:
        d = d[np.logical_not(m.walls.a)]
    except AttributeError:
        pass
    return d


def density_std(m):
    return dstd_func(density_norm(m))


def recent_dstd(dirname, t_steady):
    fnames = get_filenames(dirname)
    m_0 = filename_to_model(fnames[0])
    d_mean = np.zeros_like(m_0.c.a)
    i_samples = 0
    for fname in fnames:
        m = filename_to_model(fname)
        if m.t > t_steady:
            d_mean += density_norm(m)
            i_samples += 1
    d_mean /= float(i_samples)
    return dstd_func(d_mean)


def iterate(m, n):
    [m.iterate() for _ in range(n)]


def run_profile():
    w = walls.Walls(L=1000.0, dim=2, dx=20.0)
    m = model.Model(seed=1, dt=0.01,
                    rho_0=1e-3, v_0=20.0, D_rot=0.2,
                    p_0=1.0, chi=1e1, onesided_flag=True,
                    force_mu=0.0,
                    vicsek_R=0.0,
                    walls=w,
                    c_D=1000.0, c_sink=0.01, c_source=1.0)
    cProfile.runctx('iterate(m, n)',
                    locals={'m': m, 'n': 10000, 'iterate': iterate},
                    globals={}, sort='tottime')


def chi_dstd(dirnames):
    for dirname in dirnames:
        fnames = get_filenames(dirname)
        m_0 = filename_to_model(fnames[0])
        print(m_0.chi, recent_dstd(dirname, 5000.0))
