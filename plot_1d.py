#!/usr/bin/env python
from __future__ import print_function, division
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from runner import get_filenames, filename_to_model

if __name__ == '__main__':
    fnames = get_filenames(sys.argv[1])

    m_0 = filename_to_model(fnames[0])

    L = m_0.L

    fig = plt.figure()
    ax_vis = fig.add_subplot(311)
    ax_c = fig.add_subplot(312)
    ax_d = fig.add_subplot(313)

    ax_vis.set_xlim(-L / 2.0, L / 2.0)
    ax_c.set_xlim(-L / 2.0, L / 2.0)
    ax_d.set_xlim(-L / 2.0, L / 2.0)

    plt.subplots_adjust(left=0.25, bottom=0.25)
    plot_p = ax_vis.scatter(m_0.r[:, 0], np.zeros([m_0.n]))

    x = np.linspace(-m_0.L / 2.0, m_0.L / 2.0, m_0.c.a.shape[0])
    plot_c = ax_c.bar(x, m_0.c.a[:], width=x[1] - x[0])

    plot_d = ax_d.bar(x, m_0.get_density_field(), width=x[1] - x[0])

    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
    t_slider = Slider(ax_slide, 'Index', 0, len(fnames), valinit=0)

    def update(val):
        fname_i = int(round(val))
        if 0 <= fname_i < len(fnames):
            m = filename_to_model(fnames[fname_i])
            plot_p.set_offsets(np.array([m.r[:, 0], np.zeros([m.n])]).T)
            cs = m.c.a
            for rect, c in zip(plot_c, cs):
                rect.set_height(c)
            ax_c.set_ylim(0.0, 1.05 * cs.max())
            ds = m.get_density_field() / m.rho_0
            for rect, d in zip(plot_d, ds):
                rect.set_height(d)
            ax_d.set_ylim(0.0, 1.05 * ds.max())
            fig.canvas.draw_idle()

    t_slider.on_changed(update)

    plt.show()
