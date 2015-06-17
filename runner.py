import pickle
from os.path import join, basename, splitext, isdir
import os
import glob


def f_to_i(f):
    """Infer a model's iteration number from its output filename.

    Args:
        f (str): A path to a model output file.

    Returns:
        int: The iteration number of the model.
    """
    return int(splitext(basename(f))[0])


def get_filenames(dirname):
    """Return all model output filenames inside a model output directory,
    sorted by iteration number.

    Args:
        dirname (str): A path to a directory.

    Returns:
        list of str: Paths to all output files inside the directory, sorted in
            order of increasing iteration number.
    """
    filenames = glob.glob('{}/*.pkl'.format(dirname))
    return sorted(filenames, key=f_to_i)


def filename_to_model(filename):
    """Load a model output file and return the model.

    Args:
        filename (str): The path to a model output file.

    Returns:
        Model: The associated model instance.
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)


class Runner(object):
    """Wrapper for loading and iterating model objects, and saving their output.

    There are several ways to initialise the object.

    * If no output directory is provided, one will be automatically generated
        from the model, assuming that is provided.
    * If no model is provided, an attempt will be made to resume from the
        output directory that then is required.
    * If a model is provided, and the output directory already contains output
        from iterating the same model, the resuming/restarting behaviour depends on
        a flag.

    Args:
        output_dir (str): The path to the directory in which the model's state
            will be recorded. Potentially also the directory from which the
            model will be loaded, if a previous run is being resumed.
            If `None`, automatically generate from the model.
        output_every (int): How many iterations should occur between recording
            the model's state.
        model (Model): The model object to iterate, if a new run is being
            started.
        force_resume: A flag which is only important when there is ambiguity
            over whether to resume or restart a run: if a `model` is provided,
            but the `output_dir` already contains output files for the same
            model. In this case:
            - `True`: Resume without prompting.
            - `False`: Restart without prompting.
            - `None` (default): Prompt the user.
    """

    def __init__(self, output_dir, output_every, model=None,
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

        Args:
            n (int): Run the model for `n` iterations from its current point.
            n_upto (int): Run the model so that its iteration number is at
                least `n_upto`.
            t (float): Run the model for `t` time from its current point.
            t_upto (float): Run the model so that its time is
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
