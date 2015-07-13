from __future__ import print_function, division
import numpy as np
import model
import walls
import ramp_model
import run_utils
import defaults


def run_trap_nochi():
    extra_model_kwargs = {
        'rho_0': 1e-1,
        'onesided_flag': False,
        'chi': 0.0,
        'walls': defaults.walls_traps_1,
        'origin_flag': False,
        'p_0': 1.0,
        'c_source': 0.0,
    }
    model_kwargs = dict(defaults.default_model_2d_kwargs, **extra_model_kwargs)
    m = model.Model2D(**model_kwargs)

    output_every = 200
    t_upto = 1e4
    output_dir = None
    force_resume = None

    run_utils.run_model(output_every, output_dir, m, force_resume,
                        t_upto=t_upto)


def run_2d():
    extra_model_kwargs = {
        'rho_0': 2e-4,
        'onesided_flag': True,
        'chi': 0.0,
        'walls': defaults.walls_traps_m,
        'origin_flag': True,
    }
    model_kwargs = dict(defaults.default_model_2d_kwargs, **extra_model_kwargs)
    m = model.Model2D(**model_kwargs)

    output_every = 200
    t_upto = 1e2
    output_dir = 'test_2d'
    force_resume = None

    run_utils.run_model(output_every, output_dir, m, force_resume,
                        t_upto=t_upto)


def run_1d():
    extra_model_kwargs = {
        'rho_0': 0.1,
        'onesided_flag': True,
        'chi': 0.0,
        'origin_flag': True,
    }
    model_kwargs = dict(defaults.default_model_1d_kwargs, **extra_model_kwargs)
    m = model.Model1D(**model_kwargs)

    output_every = 200
    t_upto = 1e2
    output_dir = 'test_1d'
    force_resume = None

    run_utils.run_model(output_every, output_dir, m, force_resume,
                        t_upto=t_upto)


def run_cannock_1d():
    extra_model_kwargs = {
        'rho_0': 1.0,
        'onesided_flag': True,
        'chi': 1.5,
        'origin_flag': True,
        'vicsek_R': 0.0,
        # 'v_0': 20.0,
        'p_0': 1.0,
    }
    model_kwargs = dict(defaults.default_model_1d_kwargs, **extra_model_kwargs)
    m = model.Model1D(**model_kwargs)

    output_every = 500
    t_upto = 1e4
    output_dir = '/Users/ewj/Desktop/cannock/agent_data/{}'.format(m)
    force_resume = None

    run_utils.run_model(output_every, output_dir, m, force_resume,
                        t_upto=t_upto)


def run_chi_ramp_1d():
    ramp_kwargs = {
        'ramp_chi_0': 0.0,
        'ramp_chi_max': 10.0,
        'ramp_dchi_dt': 5e-5,
        'ramp_t_steady': 1e4,
        'ramp_dt': 32e2,
    }
    extra_model_kwargs = {
        'rho_0': 0.1,
        'onesided_flag': False,
        'chi': ramp_kwargs['ramp_chi_0'],
        'origin_flag': False,
        'vicsek_R': 10.0,
    }
    ramp_model_kwargs = dict(defaults.default_model_1d_kwargs, **ramp_kwargs)
    model_kwargs = dict(ramp_model_kwargs, **extra_model_kwargs)
    m = ramp_model.RampModel1D(**model_kwargs)

    output_every = 2000
    output_dir = None
    force_resume = None

    run_utils.run_ramp_model(output_every, output_dir, m, force_resume)


def run_chi_ramp_2d():
    ramp_kwargs = {
        'ramp_chi_0': 0.0,
        'ramp_chi_max': 10.0,
        'ramp_dchi_dt': 5e-5,
        'ramp_t_steady': 1e4,
        'ramp_dt': 32e2,
    }
    extra_model_kwargs = {
        'rho_0': 2e-4,
        'onesided_flag': True,
        # 'walls': defaults.walls_traps_1,
        'walls': defaults.walls_blank,
        'origin_flag': False,
    }
    ramp_model_kwargs = dict(defaults.default_model_2d_kwargs, **ramp_kwargs)
    model_kwargs = dict(ramp_model_kwargs, **extra_model_kwargs)
    m = ramp_model.RampModel2D(**model_kwargs)

    output_every = 2000
    output_dir = None
    force_resume = None

    run_utils.run_ramp_model(output_every, output_dir, m, force_resume)


def run_chi_scan_2d():
    extra_model_kwargs = {
        'rho_0': 1e-3,
        'onesided_flag': True,
        'walls': defaults.walls_traps_1,
        'origin_flag': True,
    }
    model_kwargs = dict(defaults.default_model_2d_kwargs, **extra_model_kwargs)

    output_every = 10000
    t_upto = 8e4
    chis = np.linspace(0.0, 800, 22)
    force_resume = True
    parallel = True

    run_utils.run_field_scan(model.Model2D, model_kwargs, output_every, t_upto,
                             'chi', chis, force_resume, parallel)


def run_trap_s_scan():
    extra_model_kwargs = {
        'rho_0': 1e-3,
        'onesided_flag': True,
        'walls': None,
        'origin_flag': True,
    }
    model_kwargs = dict(defaults.default_model_2d_kwargs, **extra_model_kwargs)
    trap_kwargs = defaults.default_trap_kwargs.copy()

    output_every = 10000
    t_upto = 8e4
    chis = np.linspace(200.0, 600.0, 11)
    force_resume = True
    parallel = True

    for s in [40.0, 80.0, 120.0, 160.0, 200.0]:
        trap_kwargs['s'] = s
        model_kwargs['walls'] = walls.Traps(**trap_kwargs)
        run_utils.run_field_scan(model.Model2D, model_kwargs, output_every,
                                 t_upto, 'chi', chis, force_resume, parallel)


def run_chi_scan_1d():
    extra_model_kwargs = {
        'rho_0': 0.5,
        'onesided_flag': True,
        'origin_flag': True,
    }
    model_kwargs = dict(defaults.default_model_1d_kwargs, **extra_model_kwargs)

    output_every = 10000
    t_upto = 4e4
    chis = np.linspace(0.0, 6.0, 28)
    force_resume = True
    parallel = True

    run_utils.run_field_scan(model.Model1D, model_kwargs, output_every, t_upto,
                             'chi', chis, force_resume, parallel)
