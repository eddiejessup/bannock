from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from runner import get_filenames, filename_to_model
from ciabatta import ejm_rcparams
import sys

ejm_rcparams.set_pretty_plots(False, False)


def plot_p(dirname):
    filenames = get_filenames(dirname)

    ts, p_mins, p_maxs, p_means = [[] for _ in range(4)]
    for filename in filenames[1:]:
        try:
            model = filename_to_model(filename)
        except:
            pass
        else:
            p_mins.append(model.p.min())
            p_maxs.append(model.p.max())
            p_means.append(model.p.mean())
            ts.append(model.t)

    plt.plot(ts, p_mins)
    plt.plot(ts, p_maxs)
    plt.plot(ts, p_means)
    plt.xlabel('Time')
    plt.ylabel('Tumble rate')
    plt.show()

if __name__ == '__main__':
    plot_p(sys.argv[1])
