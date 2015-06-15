import pickle
from os.path import join, basename, splitext, isdir
import os
import glob


def f_to_i(f):
    return int(splitext(basename(f))[0])


def get_filenames(dirname):
    filenames = glob.glob('{}/*.pkl'.format(dirname))
    return sorted(filenames, key=f_to_i)


def filename_to_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


class Runner(object):

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
        for snapshot in get_filenames(self.output_dir):
            assert snapshot.endswith('.pkl')
            os.remove(snapshot)

    def is_snapshot_time(self):
        return not self.model.i % self.output_every

    def iterate(self, n=None, n_upto=None, t=None, t_upto=None):
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
        filename = join(self.output_dir, '{:010d}.pkl'.format(self.model.i))
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def __repr__(self):
        info = '{}(out={}, model={})'
        return info.format(self.__class__.__name__, basename(self.output_dir),
                           self.model)
