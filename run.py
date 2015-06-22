import numpy as np
import model
import walls
import utils

default_wall_args = {
    'L': 5000.0,
    'dx': 40.0,
}

default_blank_args = {
    'dim': 2,
}

default_trap_args = {
    'n': 1,
    'd': 40.0,
    'w': 280.0,
    's': 80.0,
}

walls_blank = walls.Walls(**dict(default_wall_args.items() +
                                 default_blank_args.items()))
walls_traps_1 = walls.Traps(**dict(default_wall_args.items() +
                                   default_trap_args.items()))

# Make 1d default model args
default_model_1d_kwargs = {
    'seed': 1,
    'dt': 0.2,
    'v_0': 20.0,
    'p_0': 1.0,
    'origin_flag': False,
    'vicsek_R': 0.0,
    'L': default_wall_args['L'],
    'dx': default_wall_args['dx'],
    'c_D': 1000.0,
    'c_sink': 0.01,
    'c_source': 1.0,
}

# Make 2d default model args
default_model_kwargs = default_model_1d_kwargs.copy()
del default_model_kwargs['L']
del default_model_kwargs['dx']
default_model_kwargs.update({
    'D_rot': 0.2,
    'force_mu': 0.0,
})


def run_2d():
    model_kwargs = default_model_kwargs.copy()
    extra_model_kwargs = {
        'rho_0': 2e-4,
        'onesided_flag': True,
        'walls': walls_traps_1,
        'origin_flag': True,
    }
    model_kwargs.update(extra_model_kwargs)
    m = model.Model(**model_kwargs)

    output_every = 200
    t_upto = 1e2
    output_dir = 'test_2d'

    utils.run_model(output_every, model=m, output_dir=output_dir,
                    t_upto=t_upto)


def run_1d():
    model_kwargs = default_model_1d_kwargs.copy()
    extra_model_kwargs = {
        'rho_0': 0.1,
        'onesided_flag': True,
        'origin_flag': True,
    }
    model_kwargs.update(extra_model_kwargs)
    m = model.Model1D(**model_kwargs)

    output_every = 200
    t_upto = 1e2
    output_dir = 'test_1d'

    utils.run_model(output_every, m=m, output_dir=output_dir,
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
    m = model.RampModel1D(**model_kwargs)

    output_every = 2000

    utils.run_ramp_model(output_every, m=m)


def run_chi_scan_2d():
    model_kwargs = default_model_kwargs.copy()
    extra_model_kwargs = {
        'rho_0': 2e-4,
        'onesided_flag': True,
        'walls': walls_traps_1,
        'origin_flag': True,
    }
    model_kwargs.update(extra_model_kwargs)

    output_every = 5000
    t_upto = 1e4
    chis = np.linspace(250.0, 600.0, 10)

    utils.run_chi_scan_parallel(model.Model, model_kwargs, output_every,
                                t_upto, chis)


def run_chi_scan_1d():
    model_kwargs = default_model_1d_kwargs.copy()
    extra_model_kwargs = {
        'rho_0': 0.1,
        'onesided_flag': True,
        'origin_flag': True,
    }
    model_kwargs.update(extra_model_kwargs)

    output_every = 200
    t_upto = 50.0
    chis = np.linspace(2.0, 8.0, 10)

    utils.run_chi_scan(model.Model1D, model_kwargs, output_every,
                       t_upto, chis)
