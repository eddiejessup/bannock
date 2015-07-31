from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
from ciabatta import ejm_rcparams
from ciabatta.ejm_rcparams import set2
from bannock.utils import utils as butils
from cannock.utils import utils as cutils


def get_rho(m, bins):
    dx = m.L / float(bins)
    ns, bins = np.histogram(m.r[:, 0], bins=bins,
                            range=[-m.L / 2.0, m.L / 2.0])
    rho = ns / dx
    return rho


save_flag = True

use_latex = save_flag
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)

fig = plt.figure(figsize=(12, 12 * ejm_rcparams.golden_ratio))
ax = fig.add_subplot(111)
ejm_rcparams.prettify_axes(ax)

ax = fig.add_subplot(111)

dirname_noclust = '/Volumes/Backup/bannock_data/Model1D_dim=1,seed=1,dt=0.1,L=2.5e+03,dx=40,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,origin_flag=0,rho_0=0.5,chi=all,onesided_flag=1,vicsek_R=0/Model1D_dim=1,seed=1,dt=0.1,L=2.5e+03,dx=40,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,origin_flag=0,rho_0=0.5,chi=0,onesided_flag=1,vicsek_R=0'
dirname_clust = '/Volumes/Backup/bannock_data/Model1D_dim=1,seed=1,dt=0.1,L=2.5e+03,dx=40,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,origin_flag=0,rho_0=0.5,chi=all,onesided_flag=1,vicsek_R=0/Model1D_dim=1,seed=1,dt=0.1,L=2.5e+03,dx=40,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,origin_flag=0,rho_0=0.5,chi=6,onesided_flag=1,vicsek_R=0'
dirnames = [
    [r'$\chi = 0$', dirname_noclust],
    [r'$\chi = 6$', dirname_clust],
]

t_steady = 20000.0
bins = 400

cs = iter(ejm_rcparams.set2)
for label, dirname in dirnames:
    fnames = butils.get_filenames(dirname)
    ms = [butils.filename_to_model(fname) for fname in fnames]
    ms_steady = [m for m in ms if m.t > t_steady]

    m_0 = ms_steady[-1]

    rho = np.mean([get_rho(m, bins) for m in ms_steady], axis=0)
    rho_red = cutils.get_reduced_rho(m_0.rho_0, rho)

    x = np.linspace(-m_0.L / 2.0, m_0.L / 2.0, rho.shape[0])
    D_rho = cutils.get_D_rho(m_0.v_0, m_0.p_0, m_0.dim)
    x_red = cutils.get_reduced_length(m_0.c_sink, D_rho, x)

    ax.plot(x_red, rho_red, label=label, c=next(cs))

ax.legend(loc='upper left', fontsize=26)
ax.set_xlabel(r'$\tilde{x}$', fontsize=35)
ax.set_ylabel(r'$\tilde{\rho}$', fontsize=35)
ax.tick_params(axis='both', labelsize=26, pad=10.0)
ax.set_xlim(x_red.min(), x_red.max())
ax.set_ylim(0.0, None)

if save_flag:
    plt.savefig('plots/snapshot_1d.pdf', bbox_inches='tight')
else:
    plt.show()
