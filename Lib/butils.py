import pickle
import os
import numpy as np
import scipy
import obstructions

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
        # if isinstance(env.o, obstructions.Walls):
        #     stat['o'] = env.o.a
        if isinstance(env.o, obstructions.Droplet):
            stat['R'] = env.o.R
        elif isinstance(env.o, obstructions.Porous):
            stat['r'] = env.o.r
            stat['R'] = env.o.R
        return stat
    else:
        return np.load('%s/static.npz' % dirname)

def get_env(dirname):
    '''
    environment instance from data directory pickle.
    '''
    return pickle.load(open('%s/cp.pkl' % dirname, 'rb'))

def get_pos(state, skiprows=1):
    '''
    particle positions and directions from state file, either as npz dyn
    or csv
    '''
    try:
        dyn = np.load(state)
    except IOError:
        rs = np.loadtxt(state, skiprows=skiprows)
        return rs, None
    else:
        return dyn['r'], dyn['u']

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
