from __future__ import print_function, division
import utils
import model
import runner
import walls
import cProfile
import numpy as np

default_wall_args = {
    'L': 5000.0,
    'dim': 2,
    'dx': 40.0,
}


default_model_1d_kwargs = {
    'seed': 1,
    'dt': 0.2,
    'v_0': 20.0,
    'p_0': 1.0,
    'vicsek_R': 0.0,
    'L': default_wall_args['L'],
    'dx': default_wall_args['dx'],
    'c_D': 1000.0,
    'c_sink': 0.01,
    'c_source': 1.0,
}


default_model_kwargs = default_model_1d_kwargs.copy()
del default_model_kwargs['L']
del default_model_kwargs['dx']
default_model_kwargs.update({
    'D_rot': 0.2,
    'force_mu': 0.0,
})


def run_model(model_kwargs, output_dir, output_every, **iterate_args):
    m = model.Model(**model_kwargs)
    r = runner.Runner(output_dir, output_every, model=m)
    print(r.output_dir)
    r.iterate(**iterate_args)
    # cProfile.run('m.iterate(10000)' sort='tottime')
    return r


def run_model_1d(model_kwargs, output_dir, output_every, **iterate_args):
    m = model.Model1D(**model_kwargs)
    r = runner.Runner(output_dir, output_every, model=m)
    print(r.output_dir)
    r.iterate(**iterate_args)
    # cProfile.run('m.iterate(10000)' sort='tottime')
    return r


def run_ramp_model_1d(model_kwargs, output_dir, output_every):
    m = model.RampModel1D(**model_kwargs)
    r = runner.Runner(output_dir, output_every, model=m)
    print(r.output_dir)
    while r.model.chi >= 0.0:
        r.iterate(n=1)
        if r.is_snapshot_time():
            print(r.model.chi, utils.get_bcf(r.model))
    return r


def run():
    model_kwargs = default_model_kwargs.copy()
    extra_model_kwargs = {
        'rho_0': 0.1,
        'onesided_flag': False,
        'chi': 0.0,
        'vicsek_R': 10.0,
        'walls': walls.Walls(**default_wall_args),
    }
    model_kwargs.update(extra_model_kwargs)
    run_model(model_kwargs, output_dir=None, output_every=200, t_upto=2e4)


def run_1d():
    model_kwargs = default_model_1d_kwargs.copy()
    extra_model_kwargs = {
        'rho_0': 2e-5,
        'onesided_flag': False,
        'chi': 0.0,
    }
    model_kwargs.update(extra_model_kwargs)
    run_model_1d(model_kwargs, output_dir=None, output_every=200, t_upto=2e4)


def run_chi_scan_1d():
    model_kwargs = default_model_1d_kwargs.copy()
    extra_model_kwargs = {
        'rho_0': 0.1,
        'onesided_flag': False,
        'chi': 'all',
    }
    model_kwargs.update(extra_model_kwargs)

    chis = np.linspace(0.0, 20.0, 10)
    for chi in chis:
        model_kwargs['chi'] = chi
        r = run_model_1d(model_kwargs, output_dir=None, output_every=200,
                         t_upto=4e3)
        print(chi, utils.get_bcf(r.model))


def run_chi_scan():
    model_kwargs = default_model_kwargs.copy()
    extra_model_kwargs = {
        'rho_0': 2e-4,
        'onesided_flag': True,
        'chi': 'all',
        'walls': walls.Walls(**default_wall_args),
    }
    model_kwargs.update(extra_model_kwargs)

    chis = np.linspace(0.0, 40.0, 40)
    for chi in chis:
        model_kwargs['chi'] = chi
        r = run_model(model_kwargs, output_dir=None, output_every=200,
                      t_upto=1e3)
        print(chi, utils.get_bcf(r.model))


def run_chi_hysteresis_1d():
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
        'vicsek_R': 10.0,
    }
    model_kwargs = default_model_1d_kwargs.copy()
    model_kwargs.update(ramp_kwargs)
    model_kwargs.update(extra_model_kwargs)
    run_ramp_model_1d(model_kwargs, output_dir=None, output_every=2000)
