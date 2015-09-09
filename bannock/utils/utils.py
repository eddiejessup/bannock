from __future__ import print_function, division
import numpy as np
from clustrous import cluster
from agaro.measure_utils import (get_average_measure, group_by_key, t_measures,
                                 params, measures)


r_cluster_1d = 5.0
r_cluster_2d = 20.0


def _get_r_cluster(dim):
    """Find the cluster length scale appropriate for a model's dimension.

    Parameters
    ----------
    dim: int
        Dimension.

    Returns
    -------
    r_cluster: float
        Cluster length scale.
    """
    if dim == 1:
        return r_cluster_1d
    elif dim == 2:
        return r_cluster_2d


# Parameter getters


def get_chi(m):
    return m.chi


def get_time(m):
    return m.t


# Measure getters


def k_err(clust_sizes):
    c = clust_sizes
    c_err = np.sqrt(clust_sizes)
    n = float(clust_sizes.sum())
    n_err = 0.0
    n_part = n * (n - 1.0)
    n_part_err = n_err * (2.0 * n - 1.0)
    c_part_i = c * (c - 1.0)
    c_part_i_err = c_err * (2.0 * c - 1.0)
    # c_part_i_err = c_part_i_err[c_part_i_err < c_part_i_err.max()]
    c_part = np.sum(c_part_i)
    c_part_err = np.sqrt(np.sum(np.square(c_part_i_err)))
    k = c_part / n_part
    k_err = k * np.sqrt((n_part_err / n_part) ** 2 +
                        (c_part_err / c_part) ** 2)
    return k_err


def get_k(m):
    """Calculate the particle clumpiness for a model.

    Parameters
    ----------
    m: Model
        Model instance.

    Returns
    -------
    k: float
        Clumpiness measure.
    """
    labels = cluster.cluster_periodic(m.r, _get_r_cluster(m.dim), m.L)
    clust_sizes = cluster.cluster_sizes(labels)
    return cluster.clumpiness(clust_sizes), k_err(clust_sizes)


def get_fracs(m):
    """Calculate an order parameter, f, the 'confinedness',
    representing the fraction of particles in each trap for a model.

    f = 1 means all particles are in a trap.
    f = 0 means the fraction is that expected for a uniform distribution.
    f < 0 means there are fewer particles than expected.

    Parameters
    ----------
    m: Model
        Model instance.

    Returns
    -------
    fs: list[float]
        Confinedness for each trap.
    """
    fracs_0 = m.walls.get_trap_areas() / m.walls.get_free_area()
    confinednesses = (m.walls.get_fracs(m.r) - fracs_0) / (1.0 - fracs_0)
    confinedness_errs = np.zeros_like(confinednesses)
    return confinednesses, confinedness_errs


# Measures over time


def t_ks(dirname):
    """Calculate the particle clumpiness over time
    for a model output directory.

    Parameters
    ----------
    dirname: str
        A model output directory path

    Returns
    -------
    ts: numpy.ndarray[dtype=float]
        Times.
    ks: numpy.ndarray[dtype=float]
        Particle clumpinesses.
    """
    ts, ks, ks_err = t_measures(dirname, get_time, get_k)
    return ts, ks, ks_err


def t_fracs(dirname):
    """Calculate the trap confinedness over time
    for a model output directory.

    Parameters
    ----------
    dirname: str
        A model output directory path

    Returns
    -------
    ts: numpy.ndarray[dtype=float]
        Times.
    fs: numpy.ndarray[dtype=float, shape=(ts.shape[0], num_traps)]
        Trap confinednesses, where `num_traps` is the number of traps.
    """
    ts, fracs, _ = t_measures(dirname, get_time, get_fracs)
    return ts, fracs


# Parameter-measure relations


def chi_ks(dirnames, t_steady=None):
    """Calculate the particle clumpiness of a set of
    model output directories, and their associated chis.

    Parameters
    ----------
    dirnames: list[str]
        Model output directory paths.
    t_steady: None or float
        Time to consider the model to be at steady-state.
        The measure will be averaged over all later times.
        `None` means just consider the latest time.

    Returns
    -------
    chis: numpy.ndarray[dtype=float]
        Chemotactic sensitivities
    ks: numpy.ndarray[dtype=float]
        Particle clumpinesses.
    """
    chis = params(dirnames, get_chi)
    ks, ks_err = measures(dirnames, get_k, t_steady)
    return chis, ks, ks_err


def chi_fs(dirnames, t_steady=None):
    """Calculate particle confinedness of a set of
    model output directories, and their associated chis.

    Parameters
    ----------
    dirnames: list[str]
        Model output directory paths.
    t_steady: None or float
        Time to consider the model to be at steady-state.
        The measure will be averaged over all later times.
        `None` means just consider the latest time.

    Returns
    -------
    chis: numpy.ndarray[dtype=float]
        Chemotactic sensitivities
    fs: numpy.ndarray[dtype=float]
        Particle confinednesses.
    """
    chis = params(dirnames, get_chi)
    fracs, _ = measures(dirnames, get_fracs, t_steady)
    return chis, fracs


def chi_ks_run_average(dirnames, t_steady=None):
    """Calculate the particle clumpiness of a set of
    model output directories, and their associated chis.
    Take the average over all directories with equal chi.

    Parameters
    ----------
    dirnames: list[str]
        Model output directory paths.
    t_steady: None or float
        Time to consider the model to be at steady-state.
        The measure will be averaged over all later times.
        `None` means just consider the latest time.

    Returns
    -------
    chis: numpy.ndarray[dtype=float]
        Chemotactic sensitivities
    ks: numpy.ndarray[dtype=float]
        Particle clumpinesses.
    """
    chi_groups = group_by_key(dirnames, 'chi')
    chis, ks = [], []
    for chi, dirnames in chi_groups.items():
        chis.append(chi)
        ks.append(np.mean([get_average_measure(dirname, get_k, t_steady)
                           for dirname in dirnames]))
    return np.array(chis), np.array(ks)
