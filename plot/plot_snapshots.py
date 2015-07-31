from __future__ import print_function, division
import matplotlib.pyplot as plt
from ciabatta import ejm_rcparams
import numpy as np
from numpy import ma
from bannock import utils
from cannock.utils import utils as cutils


def get_r_com(r):
    return np.mean(r, axis=0)


def center_image(a, i_centre):
    a_center = np.roll(a, -i_centre[0], axis=0)
    a_center = np.roll(a_center, -i_centre[1], axis=1)
    return a_center


def vis_model(model, fig):
    ax = fig.add_subplot(111)
    ejm_rcparams.prettify_axes(ax)

    D_rho = cutils.get_D_rho(model.v_0, model.p_0, model.dim)
    L_red = cutils.get_reduced_length(delta=model.c_sink, D_rho=D_rho,
                                      x=model.L)

    blank = not model.walls.a.any()

    if blank:
        # Centre particles in the axes.
        r_com = get_r_com(model.r)
        r = model.r - r_com
        i_com = np.round(r_com / model.dx).astype(np.int)
        c = center_image(model.c.a, i_com)
    else:
        r = model.r
        mask = np.logical_not(model.walls.a)
        c = ma.array(model.c.a, mask=np.logical_not(mask), fill_value=0)
        # print(c)

    r_red = cutils.get_reduced_length(delta=model.c_sink, D_rho=D_rho,
                                      x=r)

    c_red = cutils.get_reduced_c(rho_0=model.rho_0, delta=model.c_sink,
                                 phi=model.c_source, c=c)
    c_plot = ax.imshow(c_red.T,
                       cmap=c_cmap, interpolation='nearest',
                       extent=2 * [-L_red / 2.0, L_red / 2.0], origin='lower')

    if not blank:
        walls_mask = ma.array(model.walls.a, mask=mask, fill_value=0)
        ax.imshow(walls_mask.T,
                  cmap=wall_cmap, interpolation='nearest',
                  extent=2 * [-L_red / 2.0, L_red / 2.0], origin='lower')

    ax.quiver(r_red[:, 0], r_red[:, 1],
              model.v[:, 0], model.v[:, 1])

    ax.set_xlabel(r'$\tilde{x}$', fontsize=35)
    ax.set_ylabel(r'$\tilde{y}$', fontsize=35)
    ax.tick_params(axis='both', labelsize=26, pad=10.0)
    ax.set_xlim(-L_red / 2.0, L_red / 2.0)
    ax.set_ylim(-L_red / 2.0, L_red / 2.0)
    ax.set_aspect('equal')

    c_colorbar = fig.colorbar(c_plot)
    c_colorbar.set_label(r'$\tilde{c}$', fontsize=32, rotation=0,
                         labelpad=20.0)

save_flag = True

use_latex = save_flag
use_pgf = True

ejm_rcparams.set_pretty_plots(use_latex, use_pgf)

c_cmap = ejm_rcparams.brewer2mpl.get_map('OrRd', 'sequential', 9).mpl_colormap
wall_cmap = ejm_rcparams.brewer2mpl.get_map('Blues', 'sequential', 3,
                                            reverse=True).mpl_colormap

dirnames = [
    ['blank_clust', '/Volumes/Backup/bannock_data/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.001,chi=all,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Walls_dim=2,L=2.5e+03,dx=40.3/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.001,chi=952,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Walls_dim=2,L=2.5e+03,dx=40.3'],
    ['blank_noclust', '/Volumes/Backup/bannock_data/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.001,chi=all,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Walls_dim=2,L=2.5e+03,dx=40.3/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.001,chi=0,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Walls_dim=2,L=2.5e+03,dx=40.3'],
    ['trap_noclust', '/Volumes/Backup/bannock_data/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.001,chi=all,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Traps_dim=2,L=2.5e+03,dx=40.3,n=1,d=40.3,w=282,s=80.6/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.001,chi=0,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Traps_dim=2,L=2.5e+03,dx=40.3,n=1,d=40.3,w=282,s=80.6'],
    ['trap_clust', '/Volumes/Backup/bannock_data/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.001,chi=all,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Traps_dim=2,L=2.5e+03,dx=40.3,n=1,d=40.3,w=282,s=80.6/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.001,chi=686,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Traps_dim=2,L=2.5e+03,dx=40.3,n=1,d=40.3,w=282,s=80.6'],
    ['maze_clust', '/Volumes/Backup/bannock_data/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.001,chi=all,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Maze_dim=2,L=2.5e+03,dx=40.3,d=40,seed=1/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.001,chi=800,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Maze_dim=2,L=2.5e+03,dx=40.3,d=40,seed=1'],
    ['maze_noclust', '/Volumes/Backup/bannock_data/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.001,chi=all,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Maze_dim=2,L=2.5e+03,dx=40.3,d=40,seed=1/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.001,chi=0,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Maze_dim=2,L=2.5e+03,dx=40.3,d=40,seed=1'],
    ['traps_noclust', '/Volumes/Backup/bannock_data/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.001,chi=all,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Traps_dim=2,L=2.5e+03,dx=40.3,n=5,d=40.3,w=282,s=80.6/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.001,chi=0,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Traps_dim=2,L=2.5e+03,dx=40.3,n=5,d=40.3,w=282,s=80.6'],
    ['traps_clust', '/Volumes/Backup/bannock_data/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.001,chi=all,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Traps_dim=2,L=2.5e+03,dx=40.3,n=5,d=40.3,w=282,s=80.6/Model2D_dim=2,seed=1,dt=0.1,L=2.5e+03,dx=40.3,c_D=1e+03,c_sink=0.01,c_source=1,v_0=20,p_0=1,D_rot=0.2,origin_flag=0,rho_0=0.001,chi=686,onesided_flag=1,force_mu=0,vicsek_R=0,walls=Traps_dim=2,L=2.5e+03,dx=40.3,n=5,d=40.3,w=282,s=80.6'],
][2:]

for end, dirname in dirnames:
    fnames = utils.get_filenames(dirname)
    model = utils.filename_to_model(fnames[-1])
    fig = plt.figure(figsize=(12, 12 * ejm_rcparams.golden_ratio))
    vis_model(model, fig)
    if save_flag:
        fig.savefig('plots/snapshot_{}.pdf'.format(end),
                    bbox_inches='tight')
    else:
        plt.show()
