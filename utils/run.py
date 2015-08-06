from __future__ import print_function, division
import numpy as np
from agaro.run_utils import run_model, run_kwarg_scan
from bannock import model, walls
from bannock.utils import defaults


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

    t_output_every = 20.0
    t_upto = 100.0
    output_dir = 'test_2d'
    force_resume = None

    run_model(t_output_every, output_dir, m, force_resume,
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

    t_output_every = 20.0
    t_upto = 100.0
    output_dir = 'test_1d'
    force_resume = None

    run_model(t_output_every, output_dir, m, force_resume,
              t_upto=t_upto)


def run_chi_scan_2d():
    extra_model_kwargs = {
        'rho_0': 1e-3,
        'onesided_flag': True,
        'walls': defaults.walls_blank,
        'origin_flag': False,
    }
    model_kwargs = dict(defaults.default_model_2d_kwargs, **extra_model_kwargs)

    t_output_every = 1000.0
    t_upto = 8e4
    chis = np.linspace(0.0, 800.0, 22)
    force_resume = True
    parallel = True

    model_kwarg_sets = [dict(model_kwargs, chi=chi) for chi in chis]
    run_kwarg_scan(model.Model2D, model_kwarg_sets,
                   t_output_every, t_upto, force_resume, parallel)


def run_chi_scan_1d():
    extra_model_kwargs = {
        'rho_0': 0.5,
        'onesided_flag': True,
        'origin_flag': True,
    }
    model_kwargs = dict(defaults.default_model_1d_kwargs, **extra_model_kwargs)

    t_output_every = 1000.0
    t_upto = 4e4
    chis = np.linspace(0.0, 6.0, 28)
    force_resume = True
    parallel = True

    model_kwarg_sets = [dict(model_kwargs, chi=chi) for chi in chis]
    run_kwarg_scan(model.Model1D, model_kwarg_sets,
                   t_output_every, t_upto, force_resume, parallel)


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

    t_output_every = 50.0
    t_upto = 1e4
    output_dir = '/Users/ewj/Desktop/cannock/agent_data/{}'.format(m)
    force_resume = None

    run_model(t_output_every, output_dir, m, force_resume,
              t_upto=t_upto)


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

    t_output_every = 100.0
    t_upto = 2e4
    output_dir = None
    force_resume = None

    run_model(t_output_every, output_dir, m, force_resume,
              t_upto=t_upto)


def run_trap_s_scan():
    trap_kwargs = defaults.default_trap_kwargs.copy()
    # Halve dx to let us do finer increments in `s`.
    trap_kwargs['dx'] = 20.0
    extra_model_kwargs = {
        'rho_0': 1e-3,
        'onesided_flag': True,
        'walls': None,
        'origin_flag': True,
        'dx': trap_kwargs['dx'],
        # Quarter dt to keep diffusion stable.
        'dt': 0.025,
    }
    model_kwargs = dict(defaults.default_model_2d_kwargs, **extra_model_kwargs)

    t_output_every = 2000.0
    t_upto = 16e4
    chis = np.linspace(200.0, 600.0, 22)
    force_resume = True
    parallel = True

    model_kwarg_sets = []
    for s in [20.0, 60.0, 100.0, 140.0, 180.0]:
        trap_kwargs['s'] = s
        wls = walls.Traps(**trap_kwargs)
        for chi in chis:
            model_kwargs_cur = model_kwargs.copy()
            model_kwargs_cur['walls'] = wls
            model_kwargs_cur['chis'] = chi
        model_kwarg_sets.append(model_kwargs_cur)
    run_kwarg_scan(model.Model2D, model_kwarg_sets,
                   t_output_every, t_upto, force_resume, parallel)
