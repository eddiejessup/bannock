from __future__ import print_function, division
import numpy as np
from clustrous import cluster
from agaro.output_utils import (get_recent_filename, get_filenames,
                                filename_to_model)
from agaro.measure_utils import get_average_measure, group_by_key


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
    return cluster.get_clumpiness(clust_sizes)


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
    return (m.walls.get_fracs(m.r) - fracs_0) / (1.0 - fracs_0)


def get_pstats(m):
    """Calculate the tumble rate statistics for a model.

    Parameters
    ----------
    m: Model
        Model instance.

    Returns
    -------
    p_mean: float
        Mean tumble rate, using a floor of zero, so that 0, -1 and -100
        all count as 0 when calculating the mean. This is done even if
        the model does not do this.
    p_mean: float
        Minimum tumble rate. A floor of zero is *not* used.
    p_max:float
        Maximum tumble rate.
    """
    return np.maximum(m.p, 0.0).mean(), m.p.min(), m.p.max()


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
    ts, ks = [], []
    for fname in get_filenames(dirname):
        m = filename_to_model(fname)
        ts.append(m.t)
        ks.append(get_k(m))
    return np.array(ts), np.array(ks)


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
    ts, fracs = [], []
    for fname in get_filenames(dirname):
        m = filename_to_model(fname)
        ts.append(m.t)
        fracs.append(get_fracs(m))
    return np.array(ts), np.array(fracs)


def t_pmeans(dirname):
    """Calculate tumble rates statistics over time for a model output directory.

    Parameters
    ----------
    dirname: str
        A model output directory path

    Returns
    -------
    ts: numpy.ndarray[dtype=float]
        Times.
    p_means: numpy.ndarray[dtype=float]
        Mean tumble rates, using a floor of zero, so that 0, -1 and -100 all
        count as 0 when calculating the mean. This is done even if the model
        does not do this.
    p_mins: numpy.ndarray[dtype=float]
        Minimum tumble rates. A floor of zero is *not* used.
    p_maxs: numpy.ndarray[dtype=float]
        Maximum tumble rates.
    """
    ts, p_means, p_mins, p_maxs = [], [], [], []
    for fname in get_filenames(dirname):
        m = filename_to_model(fname)
        ts.append(m.t)
        p_mean, p_min, p_max = get_pstats(m)
        p_means.append(p_mean)
        p_mins.append(p_min)
        p_maxs.append(p_max)
    return np.array(ts), np.array(p_means), np.array(p_mins), np.array(p_maxs)


def _chi_measures(dirnames, measure_func, t_steady=None):
    """Calculate a measure of a set of
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
    measures: numpy.ndarray[dtype=float]
        Measures.
    """
    chis, measures = [], []
    for dirname in dirnames:
        m_recent = filename_to_model(get_recent_filename(dirname))
        chis.append(m_recent.chi)
        measures.append(get_average_measure(dirname, measure_func, t_steady))
    return np.array(chis), np.array(measures)


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
    return _chi_measures(dirnames, get_k, t_steady)


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
    return _chi_measures(dirnames, get_fracs, t_steady)


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
