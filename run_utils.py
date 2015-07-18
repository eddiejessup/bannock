from __future__ import print_function, division
from functools import partial
import multiprocessing as mp
from ciabatta import runner
from ciabatta import runner_utils
import utils


def run_model(output_every, output_dir=None, m=None, force_resume=True,
              **iterate_args):
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
    r = runner.Runner(output_every, output_dir, m, force_resume)
    print(r.output_dir)
    r.iterate(**iterate_args)
    return r


def run_ramp_model(output_every, output_dir=None, m=None, force_resume=True):
    """Convenience function to combine making a Runner object, and
    running a RampModel until chi = 0.

    Parameters
    ----------
    m: RampModel
        Ramp model to run.
    Others:
        see :class:`Runner`.

    Returns
    -------
    r: Runner
        runner object after it has finished running for the required time.
    """
    r = runner.Runner(output_every, output_dir, m, force_resume)
    print(r.output_dir)
    while r.model.chi >= 0.0:
        r.iterate(n=1)
        if r.is_snapshot_time():
            print(r.model.chi, utils.get_k(r.model))
    return r


def _run_func(func, args, parallel=False):
    if parallel:
        mp.Pool(mp.cpu_count() - 1).map(func, args)
    else:
        for arg in args:
            func(arg)


class _TaskRunner(object):
    """Replacement for a closure, which I would use if
    the multiprocessing module supported them.

    Imagine `__init__` is the captured outside state,
    and `__call__` is the closure body.
    """

    def __init__(self, ModelClass, model_kwargs,
                 output_every, t_upto, force_resume=True):
        self.ModelClass = ModelClass
        self.model_kwargs = model_kwargs.copy()
        self.output_every = output_every
        self.t_upto = t_upto
        self.force_resume = force_resume

    def __call__(self, extra_model_kwargs):
        model_kwargs = self.model_kwargs.copy()
        model_kwargs.update(extra_model_kwargs)
        m = self.ModelClass(**model_kwargs)
        r = run_model(self.output_every, m=m, force_resume=self.force_resume,
                      t_upto=self.t_upto)
        print(extra_model_kwargs, 'k: {}'.format(utils.get_k(r.model)))


class _TaskResumeRunner(object):
    """Replacement for a closure, which I would use if
    the multiprocessing module supported them.

    Imagine `__init__` is the captured outside state,
    and `__call__` is the closure body.
    """

    def __init__(self, dirname_resume,
                 output_every, t_upto, force_resume=True):
        self.dirname_resume = dirname_resume
        self.output_every = output_every
        self.t_upto = t_upto
        self.force_resume = force_resume

    def __call__(self, extra_model_kwargs):
        # Get the model's starting state
        # from the resume directory's final state
        m = runner_utils.get_recent_model(self.dirname_resume)
        # Reset the time and number of iterations
        m.t = 0.0
        m.i = 0
        # Mark the model as being a final-state resume
        m.origin_flag = 2
        # Change the model as specified by the argument
        m.__dict__.update(extra_model_kwargs)

        r = run_model(self.output_every, m=m, force_resume=self.force_resume,
                      t_upto=self.t_upto)
        print(extra_model_kwargs, 'k: {}'.format(utils.get_k(r.model)))


def run_field_scan(ModelClass, model_kwargs, output_every, t_upto, field, vals,
                   force_resume=True, parallel=False):
    """Run many models with the same parameters but variable `field`.

    For each `val` in `vals`, a new model will be made, and run up to a time.
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
        Run each model until the time is equal to this
    field: str
        The name of the field to be varied, whose values are in `vals`.
    vals: array_like
        Iterable of values to use to instantiate each Model object.
    parallel: bool
        Whether or not to run the models in parallel, using the Multiprocessing
        library. If `True`, the number of concurrent tasks will be equal to
        one less than the number of available cores detected.
     """
    task_runner = _TaskRunner(ModelClass, model_kwargs, output_every, t_upto,
                              force_resume)
    extra_model_kwarg_sets = [{field: val} for val in vals]
    _run_func(task_runner, extra_model_kwarg_sets, parallel)


def run_field_scan_resume(dirname_resume, output_every, t_upto, field, vals,
                          force_resume=True, parallel=False):
    """Run many models with the same parameters but variable `field`.

    For each `val` in `vals`, a model will be resumed, its `field` set to `val,
    and run up to a time.
    The output directory is automatically generated from the resume model.

    Parameters
    ----------
    dirname_resume: str
        An output directory whose final output will be used to create the model
        instance.
    output_every: int
        see :class:`Runner`.
    t_upto: float
        Run each model until the time is equal to this
    field: str
        The name of the field to be varied, whose values are in `vals`.
    vals: array_like
        Iterable of values to use to instantiate each Model object.
    parallel: bool
        Whether or not to run the models in parallel, using the Multiprocessing
        library. If `True`, the number of concurrent tasks will be equal to
        one less than the number of available cores detected.
     """
    task_runner = _TaskResumeRunner(dirname_resume, output_every, t_upto,
                                    force_resume)
    extra_model_kwarg_sets = [{field: val} for val in vals]
    _run_func(task_runner, extra_model_kwarg_sets, parallel)


def resume_runs(dirnames, output_every, t_upto, parallel=False):
    """Resume many models, and run.

    Parameters
    ----------
    dirnames: list[str]
        List of output directory paths from which to resume.
    output_every: int
        see :class:`Runner`.
    t_upto: float
        Run each model until the time is equal to this
    parallel: bool
        Whether or not to run the models in parallel, using the Multiprocessing
        library. If `True`, the number of concurrent tasks will be equal to
        one less than the number of available cores detected.
     """
    run_model_partial = partial(run_model, output_every, force_resume=True,
                                t_upto=t_upto)
    _run_func(run_model_partial, dirnames, parallel)
