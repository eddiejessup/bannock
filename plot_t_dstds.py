from __future__ import print_function, division
import sys
import matplotlib.pyplot as plt
import utils

if __name__ == '__main__':
    dirname = sys.argv[1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ts, dstds = utils.t_dstds(dirname)
    plt.plot(ts, dstds)
    plt.show()
