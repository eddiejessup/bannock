import pickle
import os
import numpy as np
import scipy


def get_stat(dirname):
    '''
    Compatibility function for numpy dict
    of a data directory dirname, either from pickle
    or static numpy dict.
    '''
    if os.path.exists('%s/cp.pkl' % dirname):
        env = get_env(dirname)
        stat = {'L': env.o.L,
                'r_0': env.p.r_0}

        try:
            stat['f'] = env.f.a
        except AttributeError:
            pass
        try:
            stat['c'] = env.c.a
        except AttributeError:
            pass

        try:
            stat['o'] = env.o.a
        except AttributeError:
            pass
        try:
            stat['R'] = env.o.R
        except AttributeError:
            pass
        return stat
    else:
        return np.load('%s/static.npz' % dirname)


class PartDuck(object):
    pass


class ObsDuck(object):
    pass


class EnvDuck(object):

    def __init__(self):
        self.p = PartDuck()
        self.o = ObsDuck()


def get_env(dirname):
    '''
    environment instance from data directory pickle.
    '''
    try:
        env = pickle.load(open('%s/cp.pkl' % dirname, 'rb'))
    except IOError:
        stat = get_stat(dirname)
        env = EnvDuck()
        if 'R' in stat:
            env.p.R = stat['R']
        if 'l' in stat:
            env.p.l = stat['l']
        if 'R_d' in stat:
            env.o.R = stat['R_d']
    return env


def t(f):
    '''
    Convert a dyn filename to a time
    '''
    return float(os.path.splitext(os.path.basename(f))[0])


def R_of_l(V, l):
    '''
    radius of a spherocylinder with segment length l, such that
    total volume is V.
    '''
    R = scipy.roots([(4.0 / 3.0) * np.pi, np.pi * l, 0.0, -V])
    R_phys = np.real(R[np.logical_and(np.isreal(R), R > 0.0)])
    if len(R_phys) != 1:
        raise Exception('More or less than one physical radius found')
    return R_phys[0]


def ar(l, R):
    '''
    Aspect ratio of a spherocylinder of segment length l, radius R
    '''
    return 1.0 + l / (2.0 * R)
