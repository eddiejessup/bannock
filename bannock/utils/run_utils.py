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

    def __init__(self, t_output_every, t_upto):
        self.t_output_every = t_output_every
        self.t_upto = t_upto

    def __call__(self, dirname):
        m = runner_utils.get_recent_model(dirname)
        r = run_model(self.t_output_every, m=m,
                      force_resume=True, t_upto=self.t_upto)


def run_dirname_scan_resume(dirnames, t_output_every, t_upto,
                            parallel=False):
    """Run many models to a later time.

    For each `dirname` in `dirnames`, a model will be resumed, and run up to a
    time.

    Parameters
    ----------
    dirnames: list[str]
        Output directories to resume
    t_output_every: float
        see :class:`Runner`.
    t_upto: float
        Run each model until the time is equal to this
    parallel: bool
        Whether or not to run the models in parallel, using the Multiprocessing
        library. If `True`, the number of concurrent tasks will be equal to
        one less than the number of available cores detected.
     """
    task_runner = _TaskResumeRunner(t_output_every, t_upto)
    run_func(task_runner, dirnames, parallel)
