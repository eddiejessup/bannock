from __future__ import print_function, division
import pickle
import glob
import os
from os.path import basename, splitext
import numpy as np
from ciabatta import cluster

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


def get_bcf(m):
    """Calculate the particle clumpiness for a model.

    Parameters
    ----------
    m: Model

    Returns
    -------
    bcf: float
    """
    labels = cluster.cluster(m.r, _get_r_cluster(m))
    clust_sizes = cluster.cluster_sizes(labels)
    return cluster.cluster_measure(clust_sizes)


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


def t_bcfs(dirname):
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
    bcfs: numpy.ndarray[dtype=float]
        Particle clumpinesses.
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


def chi_bcfs(dirnames, t_steady=None):
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
    bcfs: numpy.ndarray[dtype=float]
        Particle clumpinesses.
    """
    chis, bcfs = [], []
    for dirname in dirnames:
        m_recent = filename_to_model(get_recent_filename(dirname))
        chis.append(m_recent.chi)
        bcfs.append(get_average_measure(dirname, get_bcf, t_steady))
    return np.array(chis), np.array(bcfs)


def get_average_measure(dirname, get_measure_func, t_steady=None):
    """
    Calculate a measure of a model in an output directory, averaged over
    all times when the model is at steady-state.

    Parameters
    ----------
    dirname: str
        Output directory
    get_measure_func: function
        Function which takes a :class:`Model` instance as a single argument,
        and returns the measure of interest.
    t_steady: None or float
        Time to consider the model to be at steady-state.
        `None` means just consider the latest time.

    Returns
    -------
    measure: various
        Average measure.
    """
    if t_steady is None:
        return get_measure_func(get_recent_model(dirname))
    else:
        ms = [filename_to_model(fname) for fname in get_filenames(dirname)]
        ms_steady = [m for m in ms if m.t > t_steady]
        return np.mean([get_measure_func(m) for m in ms_steady])


def format_parameter(p):
    """Format a value as a string appropriate for use in a directory name.

    For use when constructing a directory name that encodes the parameters
    of a model. Specially handled type cases are,

    - `None` is represented as 'N'.

    - `bool` is represented as '1' or '0'.

    Parameters
    ----------
    p: various

    Returns
    -------
    p_str: str
        Formatted parameter.
    """
    if isinstance(p, float):
        return '{:.3g}'.format(p)
    elif p is None:
        return 'N'
    elif isinstance(p, bool):
        return '{:d}'.format(p)
    else:
        return '{}'.format(p)


def reprify(obj, fields):
    """Make a string representing an object from a subset of its attributes.

    Parameters
    --------
    obj: object
        The object which is to be represented.
    fields: list[str]
        Strings matching the object's attributes to include in the
        representation.

    Returns
    -------
    field_strs: list[str]
        Strings, each representing a field and its value,
        formatted as '`field`=`value`'
    """
    return ['='.join([f, format_parameter(obj.__dict__[f])]) for f in fields]


def _f_to_i(f):
    """Infer a model's iteration number from its output filename.

    Parameters
    ----------
    f: str
        A path to a model output file.

    Returns
    -------
    i: int
        The iteration number of the model.
    """
    return int(splitext(basename(f))[0])


def get_filenames(dirname):
    """Return all model output filenames inside a model output directory,
    sorted by iteration number.

    Parameters
    ----------
    dirname: str
        A path to a directory.

    Returns
    -------
    filenames: list[str]
        Paths to all output files inside `dirname`, sorted in order of
        increasing iteration number.
    """
    filenames = glob.glob('{}/*.pkl'.format(dirname))
    return sorted(filenames, key=_f_to_i)


def get_recent_filename(dirname):
    """Get filename of latest-time model in a directory.
    """
    return get_filenames(dirname)[-1]


def get_recent_model(dirname):
    """Get latest-time model in a directory."""
    return filename_to_model(get_recent_filename(dirname))


def get_recent_time(dirname):
    """Get latest time in a directory."""
    return filename_to_model(get_recent_filename(dirname)).t


def get_output_every(dirname):
    """Get how many iterations between outputs have been done in a directory
    run.

    If there are multiple values used in a run, raise an exception.

    Parameters
    ----------
    dirname: str
        A path to a directory.

    Returns
    -------
    output_every: int
        The inferred number of iterations between outputs.

    Raises
    ------
    TypeError
        If there are multiple different values for `output_every` found. This
        usually means a run has been resumed with a different value.
    """
    fnames = get_filenames(dirname)
    i_s = np.array([_f_to_i(fname) for fname in fnames])
    everys = list(set(np.diff(i_s)))
    if len(everys) > 1:
        raise TypeError('Multiple values for `output_every` '
                        'found, {}.'.format(everys))
    return everys[0]


def filename_to_model(filename):
    """Load a model output file and return the model.

    Parameters
    ----------
    filename: str
        The path to a model output file.

    Returns
    -------
    m: Model
        The associated model instance.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def model_to_file(model, filename):
    """Dump a model to a file as a pickle file.

    Parameters
    ----------
    model: Model
        Model instance.
    filename: str
        A path to the file in which to store the pickle output.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def sparsify(dirname, output_every):
    """Remove files from an output directory at regular interval, so as to
    make it as if there had been more iterations between outputs. Can be used
    to reduce the storage size of a directory.

    If the new number of iterations between outputs is not an integer multiple
    of the old number, then raise an exception.

    Parameters
    ----------
    dirname: str
        A path to a directory.
    output_every: int
        Desired new number of iterations between outputs.

    Raises
    ------
    ValueError
        The directory cannot be coerced into representing `output_every`.
    """
    fnames = get_filenames(dirname)
    output_every_old = get_output_every(dirname)
    if output_every % output_every_old != 0:
        raise ValueError('Directory with output_every={} cannot be coerced to'
                         'desired new value.'.format(output_every_old))
    keep_every = output_every // output_every_old
    fnames_to_keep = fnames[::keep_every]
    fnames_to_delete = set(fnames) - set(fnames_to_keep)
    for fname in fnames_to_delete:
        os.remove(fname)
