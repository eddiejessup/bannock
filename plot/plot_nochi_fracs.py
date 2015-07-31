from __future__ import print_function, division
import matplotlib.pyplot as plt
from ciabatta import ejm_rcparams
import numpy as np
from bannock import utils as butils

save_flag = True

use_latex = save_flag
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)

fig = plt.figure(figsize=(12, 12 * ejm_rcparams.golden_ratio))
ax = fig.add_subplot(111)
ejm_rcparams.prettify_axes(ax)

dirname_align = '/Volumes/Backup/bannock_data/trap_nochi/Model2D_dim=2,seed=1,dt=0.1,L=2.48e+03,dx=40,c_D=1e+03,c_sink=0.01,c_source=0,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.1,chi=0,onesided_flag=0,force_mu=0,vicsek_R=0,walls=Traps_dim=2,L=2.48e+03,dx=40,n=1,d=40,w=280,s=120'
dirname_reflect = '/Volumes/Backup/bannock_data/trap_nochi/Model2DNoAlignment_dim=2,seed=1,dt=0.1,L=2.48e+03,dx=40,c_D=1e+03,c_sink=0.01,c_source=0,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.1,chi=0,onesided_flag=0,force_mu=0,vicsek_R=0,walls=Traps_dim=2,L=2.48e+03,dx=40,n=1,d=40,w=280,s=120'

ts_align, fracs_align = butils.t_fracs(dirname_align)
ts_reflect, fracs_reflect = butils.t_fracs(dirname_reflect)

ax.plot(ts_align, fracs_align, label='Aligning', c=ejm_rcparams.set2[0])
ax.plot(ts_reflect, fracs_reflect, label='Reflecting', c=ejm_rcparams.set2[1])
ax.axhline(color=ejm_rcparams.almost_black)

ax.legend(loc='upper left', fontsize=26)
ax.set_xlabel(r'$t / \mathrm{s}$', fontsize=35)
ax.set_ylabel(r'$f$', fontsize=35)
ax.tick_params(axis='both', labelsize=26, pad=10.0)
ax.set_xlim(0.0, None)
# ax.set_ylim(-0.01, 1.01)

if save_flag:
    plt.savefig('plots/nochi_t_fracs.pdf', bbox_inches='tight')
else:
    plt.show()
