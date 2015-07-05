from __future__ import print_function, division
import numpy as np
import model
import ramp_model
import run_utils
from defaults import (default_model_2d_kwargs, default_model_1d_kwargs,
                      walls_blank, walls_traps_1)


def run_2d():
    extra_model_kwargs = {
        'rho_0': 2e-4,
        'onesided_flag': True,
        'walls': walls_traps_1,
        'origin_flag': True,
    }
    model_kwargs = default_model_2d_kwargs.copy()
    model_kwargs.update(extra_model_kwargs)
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
        'origin_flag': True,
    }
    model_kwargs = default_model_1d_kwargs.copy()
    model_kwargs.update(extra_model_kwargs)
    m = model.Model1D(**model_kwargs)

    output_every = 200
    t_upto = 1e2
    output_dir = 'test_1d'
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
    model_kwargs = default_model_1d_kwargs.copy()
    model_kwargs.update(ramp_kwargs)
    model_kwargs.update(extra_model_kwargs)
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
        # 'walls': walls_traps_1,
        'walls': walls_blank,
        'origin_flag': False,
    }
    model_kwargs = default_model_2d_kwargs.copy()
    model_kwargs.update(ramp_kwargs)
    model_kwargs.update(extra_model_kwargs)
    m = ramp_model.RampModel2D(**model_kwargs)

    output_every = 2000
    output_dir = None
    force_resume = None

    run_utils.run_ramp_model(output_every, output_dir, m, force_resume)


def run_chi_scan_2d():
    extra_model_kwargs = {
        'rho_0': 2e-4,
        'onesided_flag': True,
        'walls': walls_traps_1,
        'origin_flag': True,
    }
    model_kwargs = default_model_2d_kwargs.copy()
    model_kwargs.update(extra_model_kwargs)

    output_every = 5000
    t_upto = 1e4
    chis = np.linspace(250.0, 600.0, 10)
    force_resume = True
    parallel = True

    run_utils.run_field_scan(model.Model2D, model_kwargs, output_every, t_upto,
                             'chi', chis, force_resume, parallel)


def run_chi_scan_1d():
    extra_model_kwargs = {
        'rho_0': 0.1,
        'onesided_flag': True,
        'origin_flag': True,
    }
    model_kwargs = default_model_1d_kwargs.copy()
    model_kwargs.update(extra_model_kwargs)

    output_every = 200
    t_upto = 50.0
    chis = np.linspace(2.0, 8.0, 10)
    force_resume = True
    parallel = True

    run_utils.run_field_scan(model.Model1D, model_kwargs, output_every, t_upto,
                             'chi', chis, force_resume, parallel)
