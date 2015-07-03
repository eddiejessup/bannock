import multiprocessing as mp
import runner
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
        r = run_model(self.output_every, m=m, t_upto=self.t_upto)
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
        Run each model until the time is equal to this
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
        Run each model until the time is equal to this
    chis: array_like[dtype=float]
        Iterable of values to use to instantiate each Model object.
     """
    task_runner = _TaskRunner(ModelClass, model_kwargs, output_every, t_upto)
    mp.Pool(mp.cpu_count() - 1).map(task_runner, chis)


def resume_runs(dirnames, output_every, t_upto):
    """Resume many models, and run.

    Parameters
    ----------
    dirnames: list[str]
        List of output directory paths from which to resume.
    output_every: int
        see :class:`Runner`.
    t_upto: float
        Run each model until the time is equal to this
     """
    for dirname in dirnames:
        run_model(output_every, output_dir=dirname, force_resume=True,
                  t_upto=t_upto)
