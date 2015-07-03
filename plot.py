from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import utils


def plot_2d(dirname):
    fig = plt.figure()
    ax_vis = fig.add_subplot(111)

    fnames = utils.get_filenames(dirname)
    m_0 = utils.filename_to_model(fnames[0])

    L = m_0.walls.L

    ax_vis.set_xlim(-L / 2.0, L / 2.0)
    ax_vis.set_ylim(-L / 2.0, L / 2.0)
    ax_vis.set_aspect('equal')

    plt.subplots_adjust(left=0.25, bottom=0.25)
    plot_p = ax_vis.quiver(m_0.r[:, 0], m_0.r[:, 1], m_0.v[:, 0], m_0.v[:, 1])

    wall_mask = np.ma.array(m_0.walls.a.T, mask=np.logical_not(m_0.walls.a.T))
    ax_vis.imshow(wall_mask, cmap='Greens_r', interpolation='nearest',
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
            m = utils.filename_to_model(fnames[fname_i])
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


def plot_1d(dirname):
    fig = plt.figure()
    ax_vis = fig.add_subplot(311)
    ax_c = fig.add_subplot(312)
    ax_d = fig.add_subplot(313)

    fnames = utils.get_filenames(dirname)

    m_0 = utils.filename_to_model(fnames[0])

    L = m_0.L

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
            m = utils.filename_to_model(fnames[fname_i])
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


def plot_t_bcfs(dirname):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ts, bcfs = utils.t_bcfs(dirname)
    ax.plot(ts, bcfs)

    plt.show()


def plot_t_dstds(dirname):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ts, dstds = utils.t_dstds(dirname)
    ax.plot(ts, dstds)

    plt.show()


def plot_t_pmeans(dirname):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ts, p_means, p_mins, p_maxs = utils.get_pmeans(dirname)
    ax.plot(ts, p_means)
    ax.plot(ts, p_mins)
    ax.plot(ts, p_maxs)

    plt.show()


def plot_chi_bcfs(dirnames):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    chis, bcfs = utils.chi_bcfs(dirnames)
    ax.scatter(chis, bcfs)
    ax.set_ylim(0.0, 1.1)

    plt.show()
