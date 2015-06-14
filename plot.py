from __future__ import print_function, division
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider
import model
import utils

fnames = sys.argv[1:]


with open(fnames[0], 'r') as f:
    m_0 = pickle.load(f)

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


def update(val):
    fname_i = int(round(val))
    if 0 <= fname_i < len(fnames):
        fname = fnames[fname_i]
        with open(fname, 'r') as f:
            m = pickle.load(f)
        plot_p.set_offsets(m.r)
        plot_p.set_UVC(m.v[:, 0], m.v[:, 1])
        c_mask = np.ma.array(np.log(m.c.a.T),
                             mask=np.logical_or(m.c.a.T == 0.0, m.walls.a.T))
        plot_c.set_data(c_mask)
        plot_c.autoscale()
        fig.canvas.draw_idle()
        print('Time: {}'.format(m.t))
        print('alpha: mean = {}, min = {}, max = {}'.format(m.p.mean(), m.p.min(), m.p.max()))
        print('c: mean = {}, min = {}, max = {}'.format(m.c.a.mean(), m.c.a.min(), m.c.a.max()))
        print('density stdev: {}'.format(utils.density_std(m)))
        print()

t_slider.on_changed(update)

plt.show()
