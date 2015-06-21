from __future__ import print_function, division
import pickle
from os.path import join, basename, splitext, isdir
import os
import glob
import multiprocessing
import utils


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
    filenames: List[str]
        Paths to all output files inside `dirname`, sorted in order of
        increasing iteration number.
    """
    filenames = glob.glob('{}/*.pkl'.format(dirname))
    return sorted(filenames, key=_f_to_i)


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
    with open(filename, 'rb') as file:
        return pickle.load(file)


class Runner(object):
    """Wrapper for loading and iterating model objects, and saving their output.

    There are several ways to initialise the object.

    - If no output directory is provided, one will be automatically generated
      from the model, assuming that is provided.

    - If no model is provided, an attempt will be made to resume from the
      output directory that then is required.

    - If a model is provided, and the output directory already contains output
      from iterating the same model, the resuming/restarting behaviour
      depends on a flag.

    Parameters
    ----------
    output_dir: str
        The path to the directory in which the model's state
        will be recorded. Potentially also the directory from which the
        model will be loaded, if a previous run is being resumed.
        If `None`, automatically generate from the model.
    output_every: int
        How many iterations should occur between recording
        the model's state.
    model: Model
        The model object to iterate, if a new run is being
        started.
    force_resume: bool
        A flag which is only important when there is ambiguity
        over whether to resume or restart a run: if a `model` is provided,
        but the `output_dir` already contains output files for the same
        model. In this case:

        - `True`: Resume without prompting.

        - `False`: Restart without prompting.

        - `None`: Prompt the user.
    """

    def __init__(self, output_every, output_dir=None, model=None,
                 force_resume=None):
        self.output_dir = output_dir
        self.output_every = output_every
        self.model = model

        if model is None and output_dir is None:
            raise ValueError('Must supply either model or directory')
        # If provided with output dir then use that
        elif output_dir is not None:
            self.output_dir = output_dir
        # If using default output dir then use that
        else:
            self.output_dir = model.__repr__()

        # If the output dir does not exist then make it
        if not isdir(self.output_dir):
            os.makedirs(self.output_dir)

        output_filenames = get_filenames(self.output_dir)

        if output_filenames:
            model_recent = filename_to_model(output_filenames[-1])

        # If a model is provided
        if model is not None:
            # Then if there is a file that contains same model as input model.
            can_resume = (output_filenames and
                          model.__repr__() == model_recent.__repr__())
            if can_resume:
                if force_resume is not None:
                    will_resume = force_resume
                else:
                    will_resume = raw_input('Resume (y/n)? ') == 'y'
                if will_resume:
                    self.model = model_recent
                else:
                    self.model = model
            else:
                self.model = model
        # If no model provided but have file from which to resume, then resume
        elif output_filenames:
            self.model = model_recent
        # If no model provided and no file from which to resume then no way
        # to get a model
        else:
            raise IOError('Cannot find any files from which to resume')

    def clear_dir(self):
        """Clear the output directory of all output files."""
        for snapshot in get_filenames(self.output_dir):
            if snapshot.endswith('.pkl'):
                os.remove(snapshot)

    def is_snapshot_time(self):
        """Determine whether or not the model's iteration number is one
        where the runner is expected to make an output snapshot.
        """
        return not self.model.i % self.output_every

    def iterate(self, n=None, n_upto=None, t=None, t_upto=None):
        """Run the model for a number of iterations, expressed in a number
        of options. Only one argument should be passed.

        Parameters
        ----------
        n: int
            Run the model for `n` iterations from its current point.
        n_upto: int
            Run the model so that its iteration number is at
            least `n_upto`.
        t: float
            Run the model for `t` time from its current point.
        t_upto: float
            Run the model so that its time is
            at least `t_upto`.
        """
        if t is not None:
            t_upto = self.model.t + t
        if t_upto is not None:
            n_upto = int(round(t_upto // self.model.dt))
        if n is not None:
            n_upto = self.model.i + n

        while self.model.i < n_upto:
            if self.is_snapshot_time():
                self.make_snapshot()
            self.model.iterate()

    def make_snapshot(self):
        """Output a snapshot of the current model state, as a pickle of the
        `Model` object in a file inside the output directory, with a name
        determined by its iteration number.
        """
        filename = join(self.output_dir, '{:010d}.pkl'.format(self.model.i))
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def __repr__(self):
        info = '{}(out={}, model={})'
        return info.format(self.__class__.__name__, basename(self.output_dir),
                           self.model)


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
    r = Runner(output_every, output_dir, m, force_resume=True)
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
    r = Runner(output_every, output_dir, m, force_resume=True)
    print(r.output_dir)
    while r.model.chi >= 0.0:
        r.iterate(n=1)
        if r.is_snapshot_time():
            print(r.model.chi, utils.get_bcf(r.model))
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
        print(chi, utils.get_bcf(r.model))


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
