from __future__ import print_function, division
import numpy as np
from runner import filename_to_model, get_filenames
from ciabatta import cluster

r_cluster_1d = 5.0
r_cluster_2d = 20.0


def get_r_cluster(m):
    if m.dim == 1:
        return r_cluster_1d
    elif m.dim == 2:
        return r_cluster_2d


def dstd_func(d):
    return np.sqrt(np.sum(np.square(d - 1.0)) / d.shape[0])


def density_norm(m):
    d = m.get_density_field() / float(m.rho_0)
    try:
        d = d[np.logical_not(m.walls.a)]
    except AttributeError:
        pass
    return d


def get_dstd_mean(dirname, t_steady):
    fnames = get_filenames(dirname)
    m_0 = filename_to_model(fnames[0])
    density_norm_mean = np.zeros_like(m_0.c.a)
    i_samples = 0
    for fname in fnames:
        m = filename_to_model(fname)
        if m.t > t_steady:
            density_norm_mean += density_norm(m)
            i_samples += 1
    density_norm_mean /= float(i_samples)
    return dstd_func(density_norm_mean)


def get_dstd(m):
    return dstd_func(density_norm(m))


def get_bcf(m):
    labels = cluster.cluster(m.r, get_r_cluster(m))
    return cluster.biggest_cluster_fraction(labels)


def get_pstats(m):
    p_mean = np.maximum(m.p, 0.0).mean()
    p_min = m.p.min()
    p_max = m.p.max()
    return np.array(p_mean), np.array(p_min), np.array(p_max)


def t_dstds(dirname):
    ts, dstds = [], []
    for fname in get_filenames(dirname):
        m = filename_to_model(fname)
        ts.append(m.t)
        dstds.append(get_dstd(m))
    return np.array(ts), np.array(dstds)


def t_bcfs(dirname):
    ts, bcfs = [], []
    for fname in get_filenames(dirname):
        m = filename_to_model(fname)
        ts.append(m.t)
        bcfs.append(get_bcf(m))
    return np.array(ts), np.array(bcfs)


def t_pmeans(dirname):
    ts, p_means, p_mins, p_maxs = [], [], [], []
    for fname in get_filenames(dirname):
        m = filename_to_model(fname)
        ts.append(m.t)
        p_mean, p_min, p_max = get_pstats(m)
        p_means.append(p_mean)
        p_mins.append(p_min)
        p_maxs.append(p_max)
    return np.array(ts), np.array(p_means), np.array(p_mins), np.array(p_maxs)


def chi_dstd(dirnames):
    chis, dstds = [], []
    for dirname in dirnames:
        fname_recent = get_filenames(dirname)[-1]
        m_recent = filename_to_model(fname_recent)
        chis.append(m_recent.chi)
        dstds.append(get_dstd(m_recent))
    return np.array(chis), np.array(dstds)


def chi_bcfs(dirnames):
    chis, bcfs = [], []
    for dirname in dirnames:
        fname_recent = get_filenames(dirname)[-1]
        m_recent = filename_to_model(fname_recent)
        chis.append(m_recent.chi)
        bcfs.append(get_bcf(m_recent))
    return np.array(chis), np.array(bcfs)


def get_hyst(dirname):
    fnames = get_filenames(dirname)
    ts, t_wraps, chis, dstds = [], [], [], []
    for fname in fnames:
        m = filename_to_model(fname)
        ts.append(m.t)
        chi, t_wrap = m.ramp_chi_func(m.t)
        chis.append(chi)
        t_wraps.append(t_wrap)
        dstds.append(get_dstd(m))
    return np.array(ts), np.array(t_wraps), np.array(chis), np.array(dstds)
