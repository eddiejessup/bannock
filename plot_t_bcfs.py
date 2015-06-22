#!/usr/bin/env python
from __future__ import print_function, division
import sys
import matplotlib.pyplot as plt
import utils

if __name__ == '__main__':
    dirname = sys.argv[1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ts, bcfs = utils.t_bcfs(dirname)
    plt.plot(ts, bcfs)
    plt.show()
