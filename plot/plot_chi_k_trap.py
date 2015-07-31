from __future__ import print_function, division
import matplotlib.pyplot as plt
from ciabatta import ejm_rcparams
import numpy as np
from glob import glob
from bannock.utils import utils

save_flag = True

use_latex = save_flag
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)

fig = plt.figure(figsize=(12, 12 * ejm_rcparams.golden_ratio))
ax = fig.add_subplot(111)
ejm_rcparams.prettify_axes(ax)

cs = iter(ejm_rcparams.set2)

dirname_sets = [
    ['One-sided, uniform', glob('/Volumes/Backup/bannock_data/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.001,chi=all,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Traps_dim=2,L=2.5e+03,dx=40.3,n=1,d=40.3,w=282,s=80.6/M*')],
    ['One-sided, origin', glob('/Volumes/Backup/bannock_data/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=1,rho_0=0.001,chi=all,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Traps_dim=2,L=2.5e+03,dx=40.3,n=1,d=40.3,w=282,s=80.6/M*')],
]

for label, dirnames in dirname_sets:
    utils.vis_chi_k(dirnames, ax, label, next(cs))

ax.legend(loc='lower right', fontsize=26)
ax.set_xlabel(r'$\tilde{\mu} / \tilde{D}_\rho$', fontsize=35)
ax.set_ylabel(r'$\kappa$', fontsize=35)
ax.tick_params(axis='both', labelsize=26, pad=10.0)
# ax.set_xlim(0.0, 16.0)
ax.set_ylim(0.0, 1.01)

if save_flag:
    plt.savefig('plots/chi_k_trap.pdf', bbox_inches='tight')
else:
    plt.show()
