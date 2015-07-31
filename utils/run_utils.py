from __future__ import print_function, division
from ciabatta import runner_utils
from agaro.run_utils import run_model
from ciabatta.parallel import run_func
from bannock.utils import utils


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
    run_func(task_runner, extra_model_kwarg_sets, parallel)
