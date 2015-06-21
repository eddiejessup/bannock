from __future__ import print_function, division
import multiprocessing
import numpy as np
import utils
import model
import runner
import walls


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


def run_model(m, output_every, output_dir=None, **iterate_args):
    r = runner.Runner(output_every, output_dir, m, force_resume=True)
    print(r.output_dir)
    r.iterate(**iterate_args)
    return r


def run_ramp_model(m, output_every, output_dir=None):
    r = runner.Runner(output_every, output_dir, m, force_resume=True)
    print(r.output_dir)
    while r.model.chi >= 0.0:
        r.iterate(n=1)
        if r.is_snapshot_time():
            print(r.model.chi, utils.get_bcf(r.model))
    return r


class _TaskRunner(object):
    """Replacement for a closure, which I would use if
    the multiprocessing module supported them.
    """

    def __init__(self, ModelClass, model_kwargs,
                 output_every, t_upto):
        self.ModelClass = ModelClass
        self.model_kwargs = model_kwargs.copy()
        self.output_every = output_every
        self.t_upto = t_upto

    def __call__(self, chi):
        self.model_kwargs['chi'] = chi
        m = self.ModelClass(**self.model_kwargs)
        r = run_model(m, output_every=self.output_every, t_upto=self.t_upto)
        print(chi, utils.get_bcf(r.model))


def run_chi_scan(ModelClass, model_kwargs, output_every, t_upto, chis):
    task_runner = _TaskRunner(ModelClass, model_kwargs, output_every, t_upto)
    multiprocessing.Pool(3).map(task_runner, chis)
    # for chi in chis:
    #     task_runner(chi)


def run():
    model_kwargs = default_model_kwargs.copy()
    extra_model_kwargs = {
        'rho_0': 2e-4,
        'onesided_flag': True,
        'walls': walls_traps_1,
    }
    model_kwargs.update(extra_model_kwargs)
    m = model.Model(**model_kwargs)
    run_model(m, output_every=200, t_upto=1e2, output_dir='test_2d')


def run_1d():
    model_kwargs = default_model_1d_kwargs.copy()
    extra_model_kwargs = {
        'rho_0': 0.1,
        'onesided_flag': True,
    }
    model_kwargs.update(extra_model_kwargs)
    m = model.Model1D(**model_kwargs)
    run_model(m, output_every=200, t_upto=1e3, output_dir='test_1d')


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
        'vicsek_R': 10.0,
    }
    model_kwargs = default_model_1d_kwargs.copy()
    model_kwargs.update(ramp_kwargs)
    model_kwargs.update(extra_model_kwargs)
    m = model.RampModel1D(**model_kwargs)
    run_ramp_model(m, output_every=2000)


def run_chi_scan_2d():
    model_kwargs = default_model_kwargs.copy()
    extra_model_kwargs = {
        'rho_0': 2e-4,
        'onesided_flag': True,
        'walls': walls_traps_1,
    }
    model_kwargs.update(extra_model_kwargs)
    run_chi_scan(model.Model, model_kwargs, output_every=400, t_upto=50.0,
                 chis=np.linspace(30.0, 75.0, 10))


def run_chi_scan_1d():
    model_kwargs = default_model_1d_kwargs.copy()
    extra_model_kwargs = {
        'rho_0': 0.1,
        'onesided_flag': True,
    }
    model_kwargs.update(extra_model_kwargs)
    run_chi_scan(model.Model1D, model_kwargs, output_every=200, t_upto=50.0,
                 chis=np.linspace(2.0, 8.0, 10))
