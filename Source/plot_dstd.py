from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from runner import get_filenames, filename_to_model
from ciabatta import ejm_rcparams
import sys

ejm_rcparams.set_pretty_plots(False, False)


def plot_p(dirname):
    filenames = get_filenames(dirname)

    ts, dstds = [[] for _ in range(2)]
    for filename in filenames[1:]:
        try:
            model = filename_to_model(filename)
        except:
            pass
        else:
            dstds.append(np.std(model.get_density_field(20.0)))
            ts.append(model.t)

    plt.plot(ts, dstds)
    plt.xlabel('Time')
    plt.ylabel('Density stdev')
    plt.show()

if __name__ == '__main__':
    plot_p(sys.argv[1])
