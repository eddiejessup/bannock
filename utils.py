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


def get_dstds(dirname):
    ts, dstds = [], []
    for fname in get_filenames(dirname):
        m = filename_to_model(fname)
        ts.append(m.t)
        dstds.append(density_std(m))
    return ts, dstds


def get_pmeans(dirname):
    ts, p_means, p_mins, p_maxs = [], [], [], []
    for fname in get_filenames(dirname):
        m = filename_to_model(fname)
        ts.append(m.t)
        p_means.append(m.p.mean())
        p_mins.append(m.p.min())
        p_maxs.append(m.p.max())
    return ts, p_means, p_mins, p_maxs


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


def run_profile_1d():
    m = model.Model1D(seed=1, dt=0.01,
                      rho_0=1e0, v_0=20.0,
                      p_0=1.0, chi=1e1, onesided_flag=True,
                      L=1000.0, dx=20.0,
                      c_D=1000.0, c_sink=0.01, c_source=1.0)
    cProfile.runctx('iterate(m, n)',
                    locals={'m': m, 'n': int(1e5), 'iterate': iterate},
                    globals={}, sort='tottime')


def chi_scan_1d():
    kwargs = {
        'seed': 1,
        'dt': 0.2,
        'rho_0': 0.1,
        'v_0': 20.0,
        'p_0': 1.0,
        'chi': None,
        'onesided_flag': False,
        'L': 5000.0,
        'dx': 40.0,
        'c_D': 1000.0,
        'c_sink': 0.01,
        'c_source': 1.0,
    }

    kwargs['chi'] = 'all'
    print(runner.make_output_dirname(kwargs))

    chis = np.linspace(0.0, 40.0, 40)
    dstds = []
    for chi in chis:
        kwargs['chi'] = chi
        m = model.Model1D(**kwargs)
        dirname = runner.make_output_dirname(kwargs)
        r = runner.Runner(output_dir=dirname, output_every=200, model=m)
        r.iterate(t_upto=2e4)

        dstd = recent_dstd(dirname, t_steady=10000.0)
        dstds.append(dstd)
        print(chi, dstd)


def chi_hysteresis_1d():
    kwargs = {
        'seed': 1,
        'dt': 0.2,
        'rho_0': 0.1,
        'v_0': 20.0,
        'p_0': 1.0,
        'chi': 0.0,
        'onesided_flag': False,
        'L': 5000.0,
        'dx': 40.0,
        'c_D': 1000.0,
        'c_sink': 0.01,
        'c_source': 1.0,
    }

    ramp_kwargs = {
        'ramp_chi_0': 2.0,
        'ramp_chi_max': 3.0,
        'ramp_dchi_dt': 5e-6,
        'ramp_t_steady': 1e4,
        'ramp_dt': 32e3,
    }

    kwargs.update(ramp_kwargs)

    dirname = runner.make_output_dirname(kwargs)
    print(dirname)
    m = model.RampModel1D(**kwargs)
    r = runner.Runner(output_dir=dirname, output_every=2000, model=m)

    while True:
        r.iterate(n=1)
        if r.is_snapshot_time():
            print(r.model.chi, density_std(r.model))


def chi_dstd(dirnames):
    for dirname in dirnames:
        fnames = get_filenames(dirname)
        m_0 = filename_to_model(fnames[0])
        print(m_0.chi, recent_dstd(dirname, 5000.0))


def hyst_data(dirname):
    fnames = get_filenames(dirname)
    ts, t_wraps, chis, dstds = [], [], [], []
    for fname in fnames:
        m = filename_to_model(fname)
        ts.append(m.t)
        chi, t_wrap = m.ramp_chi_func(m.t)
        chis.append(chi)
        t_wraps.append(t_wrap)
        dstds.append(density_std(m))
    return ts, t_wraps, chis, dstds
