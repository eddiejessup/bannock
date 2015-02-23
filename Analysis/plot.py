from __future__ import print_function, division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from runner import get_filenames, filename_to_model
from ciabatta import ejm_rcparams, utils
import sys

ejm_rcparams.set_pretty_plots(False, False)


def plot_slide(dirname):
    filenames = get_filenames(dirname)

    model_0 = filename_to_model(filenames[0])
    L = model_0.walls.L

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    plt.axis([0, 1, -10, 10])

    ax.set_xlim(-L / 2.0, L / 2.0)
    ax.set_ylim(-L / 2.0, L / 2.0)

    particle_position_plot = ax.quiver([], [], scale=100.0)
    w = model_0.walls.a
    w_m = np.ma.masked_where(np.logical_not(w), w)
    ax.imshow(w_m.T,
              extent=2*[-L / 2.0, L / 2.0], aspect='equal',
              interpolation='nearest',
              origin='lower',
              cmap=mpl.cm.binary)
    chemoattractant_plot = ax.imshow([[0]], alpha=0.5,
                                     extent=2*[-L / 2.0, L / 2.0],
                                     aspect='equal',
                                     interpolation='nearest',
                                     origin='lower',
                                     cmap=mpl.cm.Reds)
    ax_time = plt.axes([0.15, 0.1, 0.7, 0.03])

    time_slider = Slider(ax_time, 'Time', 0, len(filenames), valinit=0)

    def update(val):
        i = int(time_slider.val)
        try:
            model = filename_to_model(filenames[i])
        except:
            pass
        else:
            particle_position_plot.set_offsets(model.r)
            u = utils.vector_unit_nullnull(model.v)
            particle_position_plot.set_UVC(u[:, 0],
                                           u[:, 1])
            c = model.c.a
            c_m = np.ma.masked_where(w, c)
            chemoattractant_plot.set_data(c_m.T)
            chemoattractant_plot.autoscale()
            fig.canvas.draw()

    time_slider.on_changed(update)
    plt.show()

if __name__ == '__main__':
    plot_slide(sys.argv[1])
