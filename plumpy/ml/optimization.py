'''
'''

import sys
sys.path.insert(0, '.')

import optuna
import numpy as np
from pathlib import Path
from os import getpid
from typing import Callable

def channel_dist(ch1: int, ch2: int):
    """
    Distance function for categorical parameters channelX based on the electrode distance on the grid.

    Args:
        ch1 (int): Channel 1.
        ch2 (int): Channel 2.

    Returns:
        float: Distance on the grid (1 for adjacent electrodes, then increasing by 1 for every step).
    """
    import math
    from plumpy.utils.io import load_grid
    grid = load_grid(
        '/Fridge/bci/data/23-171_CortiCom/E_ResearchData/UBCI-CC02/7_Electrode_localization/Electrode order map/CortecGrid_electrode_label.txt')

    sqz = lambda x: np.array(x).squeeze()
    pos = lambda x: np.where(grid==x)
    p1 = sqz(pos(ch1))
    p2 = sqz(pos(ch2))
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def select_sampler(type_sampler:str, seed:int, n_features:any=None):
    """
    Select which sampler to use.

    Reference: https://optuna.readthedocs.io/en/stable/reference/samplers/index.html.
    GP and TPE seem best for multivariate problems with float and integer parameters.

    Args:
        type_sampler (str): Type of sampler to use. Choose from ['tpe', 'gp', 'bruteforce'].
        seed (int): Seed passed from the Optimizer class for deterministic results.
        n_features (int): Number of channels to optimize for. This has to be fixed for now.

    Returns:
        optuna.samplers.BaseSampler: Selected sampler.
    """
    if type_sampler == 'tpe':
        return optuna.samplers.TPESampler(seed=seed,
                                             multivariate=True,
                                             group=True,
                                             constant_liar=True,
                                             categorical_distance_func={f'channel{i}': channel_dist for i in range(n_features)})
    elif type_sampler == 'gp':
        return optuna.samplers.GPSampler(seed=seed,
                                            deterministic_objective=True)
    elif type_sampler == 'bruteforce':
        return optuna.samplers.BruteForceSampler(seed=seed)
    else:
        raise NotImplementedError

class Optimizer:
    """
    Optuna optimizer wrapper
    Args:
        args: parameters for setting up the optimizer:
                task
                subject
                model_type (=type of sampler + additional tags)
                save_path
                plot_path
                load_if_exists
        obj_fun: objective to optimize
        direction: direction to optimize for the objective: maximize or minimize
        type_sampler: type of sampler to use: gp, tpe or bruteforce
        n_features: needed for tpe to specify the categorical distance function over the channels
    """
    def __init__(self, args: dict, obj_fun: Callable, direction: str, type_sampler='gp', n_features=None) -> None:
        self.study_name = args['task'] + '_' + args['subject'] + '_' + args['model_type']
        self.save_path = Path(args['save_path']) / args['task'] / args['subject'] / args['model_type']
        self.plot_path = Path(args['plot_path']) / args['task'] / args['subject'] / args['model_type']
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.plot_path.mkdir(parents=True, exist_ok=True)
        self.n_trials = args['n_trials']
        self.objective = obj_fun
        self.direction = direction
        self.type_sampler = type_sampler
        self.n_features = n_features
        self.load_if_exists = args['load_if_exists']
        print(self.save_path)

        if args['load_if_exists'] == False:
            try:
                optuna.study.delete_study(self.study_name, storage='sqlite:///' + str(self.save_path / self.study_name) + '.db')
            except Exception as e:
                pass

        seed = np.random.choice(1000, 1) # cannot set constant seed because of parallel distributed setup
        sampler = select_sampler(type_sampler, seed, n_features)
        np.savetxt(str(self.save_path / 'sampler_seed_process_') + str(getpid()) + '.txt', seed, fmt='%4d') # save seed to reproduce results
        self.study = optuna.create_study(study_name=self.study_name,
                                    sampler=sampler, storage='sqlite:///' + str(self.save_path / self.study_name) + '.db',
                                    direction=direction,
                                    load_if_exists=self.load_if_exists)
    def enquire(self, params):
        self.study.enqueue_trial(params)

    def optimize(self):
        self.study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1, gc_after_trial=True)
        self.study.trials_dataframe().to_csv(str(self.save_path / 'optuna_trials.tsv'), sep='\t', index=False)
