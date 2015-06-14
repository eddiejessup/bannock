from __future__ import print_function, division
import numpy as np
from runner import filename_to_model, get_filenames
from ciabatta import cluster

r_cluster = 40.0


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
        p_means.append(np.maximum(m.p, 0.0).mean())
        p_mins.append(m.p.min())
        p_maxs.append(m.p.max())
    return ts, p_means, p_mins, p_maxs


def get_big_cluster_fractions(dirname):
    ts, bcfs = [], []
    for fname in get_filenames(dirname)[::10]:
        m = filename_to_model(fname)
        ts.append(m.t)
        labels = cluster.cluster(m.r, r_cluster)
        bcf = cluster.biggest_cluster_fraction(labels)
        bcfs.append(bcf)
    return ts, bcfs


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
