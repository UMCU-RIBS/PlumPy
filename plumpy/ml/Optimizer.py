'''
'''

import sys
sys.path.insert(0, '.')

import multiprocessing
import optuna
import json
import numpy as np
import sys
sys.path.insert(0, '/home/julia/Documents/Python/RiverFErn')
from pathlib import Path
from os import getpid
from optuna import Trial
from contextlib import contextmanager
from riverfern.utils.plots import plot_optuna_opt_history, plot_optuna_opt_param

N_GPUS = 1

class GpuQueue:

    def __init__(self):
        self.queue = multiprocessing.Manager().Queue()
        all_idxs = list(range(N_GPUS)) if N_GPUS > 0 else [None]
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)


# class Objective:
#
#     def __init__(self, obj_fun: any, gpu_queue=None):
#         self.gpu_queue = gpu_queue
#         self.obj_fun = obj_fun
#
#     def __call__(self, trial: Trial, ):
#         params = {
#             'weights': np.array([trial.suggest_float('weights', -10., 10., step=.1) for i in self.args['n_weights']]),
#         }
#
#         params['trial_id'] = trial.number
#         obj_val = self.obj_fun(params)
#
#         return obj_val

class Optimizer:
    def __init__(self, args: dict) -> None:
        self.study_name = args['task'] + '_' + args['subject'] + '_' + args['model_type']
        self.save_path = Path(args['save_path']) / args['task'] / args['subject'] / args['model_type']
        self.plot_path = Path(args['plot_path']) / args['task'] / args['subject'] / args['model_type']
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.plot_path.mkdir(parents=True, exist_ok=True)
        self.n_trials = args['n_trials']
        print(self.save_path)

        if args['load_if_exists'] == False:
            try:
                optuna.study.delete_study(self.study_name, storage='sqlite:///' + str(self.save_path / self.study_name) + '.db')
            except Exception as e:
                pass
        self.load_if_exists = args['load_if_exists']

    def optimize(self, obj_fun, directions):
        seed = np.random.choice(1000, 1) # cannot set constant seed because of parallel distributed setup
        sampler = optuna.samplers.TPESampler(seed=seed)  # Make the sampler behave in a deterministic way. seed=78
        np.savetxt(str(self.save_path / 'sampler_seed_process_') + str(getpid()) + '.txt', seed, fmt='%4d') # save seed to reproduce results
        self.study = optuna.create_study(study_name=self.study_name,
                                    sampler=sampler, storage='sqlite:///' + str(self.save_path / self.study_name) + '.db',
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=10,
                                                                       n_warmup_steps=5,
                                                                       interval_steps=10,
                                                                       n_min_trials=50),
                                    directions=directions,
                                    load_if_exists=self.load_if_exists)

        self.study.optimize(obj_fun, n_trials=self.n_trials, n_jobs=1)
        self.study.trials_dataframe().to_csv(str(self.save_path / 'optuna_trials.tsv'), sep='\t')

        # fig = optuna.visualization.plot_param_importances(self.study)
        # fig.write_image(str(self.plot_path / 'importances'), format='pdf')
        # plot_optuna_opt_history(self.study, self.plot_path)
        #
        # fig = optuna.visualization.plot_contour(self.study, params=[fig['data'][0].y[-1], fig['data'][0].y[-2]])
        # fig.write_image(str(self.plot_path / 'contour'), format='pdf')
        #
        # fig = optuna.visualization.plot_optimization_history(self.study)
        # fig.write_image(str(self.plot_path / 'opt_history'), format='pdf')
        #
        # fig = optuna.visualization.plot_slice(self.study)
        # fig.write_image(str(self.plot_path / 'param_slice'), format='pdf')

        # json.dump(self.study.best_params,
        #           open(str(self.save_path / 'best_params.json'), 'w'),
        #           indent=4,
        #           sort_keys=True)
        #
        # json.dump(self.study.best_trial.__dict__,
        #           open(str(self.save_path / 'best_trial.json'), 'w'),
        #           indent=4,
        #           sort_keys=True,
        #           default=str)
        #
        # print('Best parameters')
        # for key, value in self.study.best_params.items():
        #     print(key + '=' + str(value) + '\n')
