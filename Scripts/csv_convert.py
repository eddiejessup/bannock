import numpy as np
import glob
import os

dirname = '/Users/ejm/Projects/Bannock/Data/drop/exp/xyz_wholedrop_new_filt'

for fname in glob.glob('/Users/ejm/Projects/Bannock/Data/drop/exp/xyz_wholedrop_new/*.csv'):
    slices, xs, ys, zs = np.genfromtxt(fname, delimiter=',', unpack=True, skiprows=1)

    fname_new_even = '{}_o.csv'.format(os.path.splitext(os.path.basename(fname))[0])
    fname_new_odd = '{}_e.csv'.format(os.path.splitext(os.path.basename(fname))[0])
    print(fname)
    e = slices % 2 == 0
    o = slices % 2 == 1

    np.savetxt(os.path.join(dirname, fname_new_odd), list(zip(xs[o], ys[o], zs[o])))
    np.savetxt(os.path.join(dirname, fname_new_even), list(zip(xs[e], ys[e], zs[e])))
