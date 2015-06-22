#!/usrbin/env python
from __future__ import print_function, division
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from runner import get_filenames, filename_to_model
import utils

if __name__ == '__main__':
    fnames = get_filenames(sys.argv[1])

    m_0 = filename_to_model(fnames[0])

    L = m_0.walls.L

    fig = plt.figure()
    ax_vis = fig.add_subplot(111)
    ax_vis.set_xlim(-L / 2.0, L / 2.0)
    ax_vis.set_ylim(-L / 2.0, L / 2.0)
    ax_vis.set_aspect('equal')

    plt.subplots_adjust(left=0.25, bottom=0.25)
    plot_p = ax_vis.quiver(m_0.r[:, 0], m_0.r[:, 1], m_0.v[:, 0], m_0.v[:, 1])

    wall_mask = np.ma.array(m_0.walls.a.T, mask=np.logical_not(m_0.walls.a.T))
    plot_w = ax_vis.imshow(wall_mask, cmap='Greens_r', interpolation='nearest',
                           origin='lower', extent=2 * [-L / 2.0, L / 2.0])

    plot_c = ax_vis.imshow([[0]], cmap='Reds', interpolation='nearest',
                           origin='lower', extent=2 * [-L / 2.0, L / 2.0])

    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
    t_slider = Slider(ax_slide, 'Time', 0, len(fnames), valinit=0)

    ax_cb = plt.axes([0.875, 0.2, 0.05, 0.7])
    fig.colorbar(plot_c, cax=ax_cb)

    t_time = fig.text(0.1, 0.5, '')
    t_dstd = fig.text(0.1, 0.4, '')

    def update(val):
        fname_i = int(round(val))
        if 0 <= fname_i < len(fnames):
            m = filename_to_model(fnames[fname_i])
            plot_p.set_offsets(m.r)
            plot_p.set_UVC(m.v[:, 0], m.v[:, 1])
            c_mask = np.ma.array(np.log(m.c.a.T),
                                 mask=np.logical_or(m.c.a.T == 0.0,
                                                    m.walls.a.T))
            plot_c.set_data(c_mask)
            plot_c.autoscale()
            t_time.set_text('Time: {:g}'.format(m.t))
            t_dstd.set_text('Dstd: {:g}'.format(utils.get_dstd(m)))
            fig.canvas.draw_idle()

    t_slider.on_changed(update)

    plt.show()
