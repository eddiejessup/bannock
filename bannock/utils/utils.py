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
    """Calculate the fraction of particles inside each trap.

    Parameters
    ----------
    r: array_like[shape=(n, 2)]
        Particle position vectors.

    Returns
    -------
    fracs: list[int]
        Fraction of the total population that is inside each trap.
    """
    inds = m.walls.r_to_i(m.r)
    n_traps = [0 for i in range(len(m.walls.traps_i))]
    w_i_half = m.walls.w_i // 2
    for i_trap in range(len(m.walls.traps_i)):
        mid_x, mid_y = m.walls.traps_i[i_trap]

        low_x, high_x = mid_x - w_i_half, mid_x + w_i_half
        low_y, high_y = mid_y - w_i_half, mid_y + w_i_half
        for i_x, i_y in inds:
            if low_x <= i_x <= high_x and low_y <= i_y <= high_y:
                n_traps[i_trap] += 1
    n_traps = np.array(n_traps)
    n_traps_err = np.sqrt(n_traps)
    fracs = n_traps / m.n
    fracs_err = n_traps_err / m.n
    return fracs, fracs_err


def get_conf(m):
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
    fracs_0 = m.walls.get_trap_areas() / m.walls.free_area
    fracs, fracs_err = get_fracs(m)
    confs = (fracs - fracs_0) / (1.0 - fracs_0)
    print(fracs_0)
    conf_errs = fracs_err / (1.0 - fracs_0)
    return confs, conf_errs


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


def t_confs(dirname):
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
    ts, confs, confs_err = t_measures(dirname, get_time, get_conf)
    return ts, confs, confs_err


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


def chi_confs(dirnames, t_steady=None):
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
    confs, confs_err = measures(dirnames, get_conf, t_steady)
    return chis, confs, confs_err


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
