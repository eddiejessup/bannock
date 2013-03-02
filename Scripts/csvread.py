#! /usr/bin/python3
import sys
import numpy as np
import matplotlib.mlab as mlb

fname = sys.argv[1]
d = mlb.csv2rec(fname, delimiter=' ')
t_0 = float(sys.argv[3])
field = sys.argv[2]
samples = d[field][np.where(d['t'] > t_0)]

n = len(samples)
mean = np.mean(samples)
stderr = np.std(samples) / np.sqrt(n)
fracerr = np.abs(stderr / mean)
try:
    if sys.argv[4] == '-v':
        print('%s: %f\u00B1%f (%.1f%% error), t_start: %g' % (field, mean, stderr, 100.0 * fracerr, t_0))
    else:
        raise Exception
except:
    print('%f %f' % (mean, stderr))
