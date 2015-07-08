from __future__ import print_function, division
import walls

default_wall_args = {
    'L': 5000.0,
    'dx': 40.0,
}

default_extra_blank_args = {
    'dim': 2,
}
default_blank_args = default_wall_args.copy()
default_blank_args.update(default_extra_blank_args)

default_extra_trap_args = {
    'n': 1,
    'd': 40.0,
    'w': 280.0,
    's': 80.0,
}
default_trap_args = default_wall_args.copy()
default_trap_args.update(default_extra_trap_args)

walls_blank = walls.Walls(**default_blank_args)
walls_traps_1 = walls.Traps(**default_trap_args)

# Make 1d default model args
default_model_1d_kwargs = {
    'seed': 1,
    'dt': 0.1,
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
default_model_2d_kwargs = default_model_1d_kwargs.copy()
del default_model_2d_kwargs['L']
del default_model_2d_kwargs['dx']
default_model_2d_kwargs.update({
    'D_rot': 0.2,
    'force_mu': 0.0,
})
