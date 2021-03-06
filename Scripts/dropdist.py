#! /usr/bin/env python3


import argparse
import numpy as np
import utils
import droplyse
import scipy.stats
import glob
import os


def rdf(dirname, bins, res, theta_max):
    fnames = glob.glob('{}/dyn/*.npz'.format(dirname))
    Rs_edges, R_drops, n_s, ns_s = [], [], [], []
    for i_d, fname in enumerate(fnames):
        if not droplyse.is_hemisphere(fname):
            pre, ext = os.path.splitext(os.path.basename(fname))
            t = float(pre)
            if t < 20.0:
                continue

        xyz, R_drop, hemisphere = droplyse.parse(fname, theta_max)
        n = len(xyz)
        r = utils.vector_mag(xyz)

        if res is not None:
            bins = droplyse.res_to_bin(droplyse.buff * R_drop, res)
        ns, R_edges = np.histogram(r, bins=bins, range=[0.0, droplyse.buff * R_drop])

        Rs_edges.append(R_edges)
        R_drops.append(R_drop)
        ns_s.append(ns)
        n_s.append(n)

    R_edges = np.mean(Rs_edges, axis=0)
    R_drop = np.mean(R_drops)
    ns = np.mean(ns_s, axis=0)
    ns_err = scipy.stats.sem(ns_s, axis=0)
    n = np.mean(n_s)

    Vs_edge, rhos = droplyse.n_to_rho(R_edges, ns, droplyse.dim, hemisphere, theta_max)
    Vs_edge, rhos_err = droplyse.n_to_rho(R_edges, ns_err, droplyse.dim, hemisphere, theta_max)
    rho_0 = droplyse.n0_to_rho0(n, R_drop, droplyse.dim, hemisphere, theta_max)

    Rs = 0.5 * (R_edges[:-1] + R_edges[1:])

    Rs_norm = Rs / R_drop
    rhos_norm = rhos / rho_0
    rhos_norm_err = rhos_err / rho_0
    return Rs_norm, rhos_norm, rhos_norm_err


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyse droplet distributions')
    parser.add_argument('dir',
                        help='Data directory')
    parser.add_argument('-b', '--bins', type=int,
                        help='Number of bins to use')
    parser.add_argument('-r', '--res', type=float,
                        help='Bin resolution in micrometres')
    parser.add_argument('-t', '--theta_factor', type=float, default=2.0,
                        help='Solid angle in reciprocal factor of pi')
    parser.add_argument('-o', '--out',
                        help='Output file')

    args = parser.parse_args()

    if args.out is None:
        args.out = args.dir + '_rdf.txt'

    theta_max = np.pi / args.theta_factor
    Rs_norm, rhos_norm, rhos_norm_err = rdf(args.dir, args.bins, args.res, theta_max)
    np.savetxt(args.out, list(zip(Rs_norm, rhos_norm, rhos_norm_err)), header='R_norm rho_norm rho_norm_err')
