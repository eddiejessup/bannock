from __future__ import print_function, division
import multiprocessing
import numpy as np
from ciabatta import cluster
from runner import filename_to_model, get_filenames
import runner

r_cluster_1d = 5.0
r_cluster_2d = 20.0


def _get_r_cluster(m):
    """Find the cluster length scale appropriate for a model's dimension.

    Parameters
    ----------
    m: Model

    Returns
    -------
    r_cluster: float
    """
    if m.dim == 1:
        return r_cluster_1d
    elif m.dim == 2:
        return r_cluster_2d


def dstd_func(d):
    """Calculate the heterogeneity of a normalised density field.

    For a normalised density field, that is, an array with mean 1, calculate
    its deviation from uniformity, that is, 1 everywhere, computed like a
    standard deviation.

    Parameters
    ----------
    d: numpy.ndarray[dtype=float]
        Normalised density field.

    Returns
    -------
    dstd: float
    """
    return np.sqrt(np.sum(np.square(d - 1.0)) / d.shape[0])


def _density_norm(m):
    """Calculate a model's normalised density field.

    Calculate a model's density field, normalise it by the mean density,
    and ignore points that are on obstacles.

    Parameters
    ----------
    m: Model

    Returns
    -------
    d: numpy.ndarray[dtype=float]
    """
    d = m.get_density_field() / float(m.rho_0)
    try:
        d = d[np.logical_not(m.walls.a)]
    except AttributeError:
        pass
    return d


def get_dstd_mean(dirname, t_steady):
    """Return the density field heterogeneity of a model's output, averaged
    over many output times after steady state is reached.

    Parameters
    ----------
    dirname: str
        A model output directory name
    t_steady: float
        The time at which to consider the model to be at steady state.
        Averaging is done over all later times.

    Returns
    -------
    dstd: float
        The heterogeneity measure.
    """
    fnames = get_filenames(dirname)
    m_0 = filename_to_model(fnames[0])
    density_norm_mean = np.zeros_like(m_0.c.a)
    i_samples = 0
    for fname in fnames:
        m = filename_to_model(fname)
        if m.t > t_steady:
            density_norm_mean += _density_norm(m)
            i_samples += 1
    density_norm_mean /= float(i_samples)
    return dstd_func(density_norm_mean)


def get_dstd(m):
    """Calculate the density field heterogeneity of a model.

    Parameters
    ----------
    m: Model

    Returns
    -------
    dstd: float
        The heterogeneity measure.
    """
    return dstd_func(_density_norm(m))


def get_bcf(m):
    """Calculate the fraction of particles in the biggest cluster for a model.

    Parameters
    ----------
    m: Model

    Returns
    -------
    bcf: float
    """
    labels = cluster.cluster(m.r, _get_r_cluster(m))
    return cluster.biggest_cluster_fraction(labels)


def get_pstats(m):
    """Calculate the tumble rate statistics for a model.

    Parameters
    ----------
    m: Model

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


def t_dstds(dirname):
    """Calculate the density field heterogeneity over time for a model output
    directory.

    Parameters
    ----------
    dirname: str
        A model output directory path.

    Returns
    -------
    ts: numpy.ndarray[dtype=float]
        Times.
    dstds: numpy.ndarray[dtype=float]
        Heterogeneity measures.
    """
    ts, dstds = [], []
    for fname in get_filenames(dirname):
        m = filename_to_model(fname)
        ts.append(m.t)
        dstds.append(get_dstd(m))
    return np.array(ts), np.array(dstds)


def t_bcfs(dirname):
    """Calculate the fraction of particles in the biggest cluster over time
    for a model output directory.

    Parameters
    ----------
    dirname: str
        A model output directory path

    Returns
    -------
    ts: numpy.ndarray[dtype=float]
        Times.
    bcfs: numpy.ndarray[dtype=float]
        Biggest cluster fractions.
    """
    ts, bcfs = [], []
    for fname in get_filenames(dirname):
        m = filename_to_model(fname)
        ts.append(m.t)
        bcfs.append(get_bcf(m))
    return np.array(ts), np.array(bcfs)


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


def chi_dstd(dirnames):
    """Calculate the density field heterogeneity of a set of model output
    directories, and their associated chis.

    Parameters
    ----------
    dirnames: List[str]
        Model output directory paths.

    Returns
    -------
    chis: numpy.ndarray[dtype=float]
        Chemotactic sensitivities
    dstds: numpy.ndarray[dtype=float]
        Heterogeneity measures.
    """
    chis, dstds = [], []
    for dirname in dirnames:
        fname_recent = get_filenames(dirname)[-1]
        m_recent = filename_to_model(fname_recent)
        chis.append(m_recent.chi)
        dstds.append(get_dstd(m_recent))
    return np.array(chis), np.array(dstds)


def chi_bcfs(dirnames):
    """Calculate the fraction of particles in the biggest cluster of a set of
    model output directories, and their associated chis.

    Parameters
    ----------
    dirnames: List[str]
        Model output directory paths.

    Returns
    -------
    chis: numpy.ndarray[dtype=float]
        Chemotactic sensitivities
    bcfs: numpy.ndarray[dtype=float]
        Biggest cluster fractions.
    """
    chis, bcfs = [], []
    for dirname in dirnames:
        fname_recent = get_filenames(dirname)[-1]
        m_recent = filename_to_model(fname_recent)
        chis.append(m_recent.chi)
        bcfs.append(get_bcf(m_recent))
    return np.array(chis), np.array(bcfs)


def chi_dstd_ramp(dirname):
    """Calculate the density field heterogeneity over time for a model output
    directory, under a ramping protocol for chi.

    Parameters
    ----------
    dirname: str
        Model output directory path.

    Returns
    -------
    ts: numpy.ndarray[dtype=float]
        Times.
    ts_wrap: numpy.ndarray[dtype=float]
        Times, reflected about the maximum chi when the ramp starts decreasing.
        Use these to make a hysteresis plot.
    chis: numpy.ndarray[dtype=float]
    dstds: numpy.ndarray[dtype=float]
        Heterogeneity measures.
    """
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


def run_model(m, output_every, output_dir=None, **iterate_args):
    """Convenience function to combine making a Runner object, and
    running it for some time.

    Parameters
    ----------
    m: Model
        Model to run.
    iterate_args:
        Arguments to pass to :meth:`Runner.iterate`.
    Others:
        see :class:`Runner`.

    Returns
    -------
    r: Runner
        runner object after it has finished running for the required time.
    """
    r = runner.Runner(output_every, output_dir, m, force_resume=True)
    print(r.output_dir)
    r.iterate(**iterate_args)
    return r


def run_ramp_model(m, output_every, output_dir=None):
    """Convenience function to combine making a Runner object, and
    running a RampModel until chi = 0.

    Parameters
    ----------
    m: RampModel1D
        Ramp model to run.
    Others:
        see :class:`Runner`.

    Returns
    -------
    r: Runner
        runner object after it has finished running for the required time.
    """
    r = runner.Runner(output_every, output_dir, m, force_resume=True)
    print(r.output_dir)
    while r.model.chi >= 0.0:
        r.iterate(n=1)
        if r.is_snapshot_time():
            print(r.model.chi, get_bcf(r.model))
    return r


class _TaskRunner(object):
    """Replacement for a closure, which I would use if
    the multiprocessing module supported them.
    """

    def __init__(self, ModelClass, model_kwargs,
                 output_every, t_upto):
        self.ModelClass = ModelClass
        self.model_kwargs = model_kwargs.copy()
        self.output_every = output_every
        self.t_upto = t_upto

    def __call__(self, chi):
        self.model_kwargs['chi'] = chi
        m = self.ModelClass(**self.model_kwargs)
        r = run_model(m, output_every=self.output_every, t_upto=self.t_upto)
        print(chi, get_bcf(r.model))


def run_chi_scan(ModelClass, model_kwargs, output_every, t_upto, chis):
    """Run many models with the same parameters but variable chi.

    For each `chi` in `chis`, a new model will be made, and run up to a time.
    The output directory is automatically generated from the model arguments.

    Parameters
    ----------
    ModelClass: type
        A class that can be instantiated into a Model object by calling
        `ModelClass(model_kwargs)`
    model_kwargs: dict
        Arguments that can instantiate a `ModelClass` object when passed
        to the `__init__` method.
    output_every: int
        see :class:`Runner`.
    t_upto: float
        How long to run each model for
    chis: array_like[dtype=float]
        Iterable of values to use to instantiate each Model object.
     """
    task_runner = _TaskRunner(ModelClass, model_kwargs, output_every, t_upto)
    for chi in chis:
        task_runner(chi)


def run_chi_scan_parallel(ModelClass, model_kwargs, output_every, t_upto,
                          chis):
    """Run many models with the same parameters but variable chi.

    Run them in parallel using the Multiprocessing library.

    For each `chi` in `chis`, a new model will be made, and run up to a time.
    The output directory is automatically generated from the model arguments.

    Parameters
    ----------
    ModelClass: type
        A class that can be instantiated into a Model object by calling
        `ModelClass(model_kwargs)`
    model_kwargs: dict
        Arguments that can instantiate a `ModelClass` object when passed
        to the `__init__` method.
    output_every: int
        see :class:`Runner`.
    t_upto: float
        How long to run each model for
    chis: array_like[dtype=float]
        Iterable of values to use to instantiate each Model object.
     """
    task_runner = _TaskRunner(ModelClass, model_kwargs, output_every, t_upto)
    multiprocessing.Pool(3).map(task_runner, chis)
