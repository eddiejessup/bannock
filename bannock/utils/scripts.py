from __future__ import print_function, division
import sys
import matplotlib.pyplot as plt
from bannock.plot import plot


def plot_chi_fs():
    ax = plt.gca()
    plot.plot_chi_fs(sys.argv[1:], ax)
    plt.show()


def plot_chi_ks():
    ax = plt.gca()
    plot.plot_chi_ks(sys.argv[1:], ax)
    plt.show()


def plot_t_confs():
    ax = plt.gca()
    plot.plot_t_confs(sys.argv[1], ax)
    plt.show()


def plot_t_ks():
    ax = plt.gca()
    plot.plot_t_ks(sys.argv[1], ax)
    plt.show()


def plot_vis():
    plot.plot_vis(sys.argv[1])
