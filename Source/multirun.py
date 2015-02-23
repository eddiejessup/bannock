import multiprocessing
from functools import partial
from runner import Runner, make_output_dirname, get_filenames
from model import Model
from os.path import join
from parameters import defaults
from itertools import product


def iterate(runner, t_upto):
    print(runner.model)
    runner.iterate(t_upto=t_upto)


def pool_run(runners, t_upto):
    iterate_partial = partial(iterate, t_upto=t_upto)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    pool.map_async(iterate_partial, runners).get(1e100)
    pool.close()
    pool.join()


def run_param_sweep(super_dirname, output_every, t_upto, resume, walls,
                    chis, force_mus, vicsek_Rs, seeds,
                    **kwargs):
    args = defaults.copy()
    args.update(kwargs)
    args['walls'] = walls

    if isinstance(chis, float):
        chis = [chis]
    if isinstance(force_mus, float):
        force_mus = [force_mus]
    if isinstance(vicsek_Rs, float):
        vicsek_Rs = [vicsek_Rs]
    if isinstance(seeds, int):
        seeds = [seeds]

    runners = []
    for chi, force_mu, vicsek_R, seed in product(chis, force_mus,
                                                 vicsek_Rs, seeds):
        args['seed'] = seed
        args['chi'] = chi
        args['force_mu'] = force_mu
        args['vicsek_R'] = vicsek_R

        output_dirname = make_output_dirname(args)
        output_dirpath = join(super_dirname, output_dirname)
        if resume and get_filenames(output_dirpath):
            runner = Runner(output_dirpath, output_every)
        else:
            model = Model(**args)
            runner = Runner(output_dirpath, output_every, model=model)
            runner.clear_dir()
        runners.append(runner)

    pool_run(runners, t_upto)
