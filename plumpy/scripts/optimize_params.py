'''
SETUP:
    virtual envirornment with the matlab engine: matlab (miniconda)
    had to be a separate env because current matlab version is 23.2.1, it is only compatible with python < 3.12
    so I ran:

    conda create -n matlab python=3.10
    python -m pip install matlabengine==23.2.1

    prior to this, followed instructions here: https://pypi.org/project/matlabengine/

USE:
    python /home/julia/Documents/Python/PlumPy/plumpy/scripts/optimize_params.py

    Needs .yml with parameters:
        /Fridge/users/julia/project_corticom/cc2/config_opt_matlab.yml
    Can monitor results in real time using:
        optuna-dashboard sqlite:///pathToDB


below needs to be better organized. There are different scenarios I have tried:
0) TPE sampler without multivariate=True and group=True, all channels, multiobjective: maximize TP, minimize FP.
    This led to bad results for both grasp and selecteer:
                    optimize_mat_multiclicks2 and optimize_mat_multiclicks3
                    Optimizer
1) TPE with multivariate=True and group=True, only upper 2 grids for grasp and bottom 2 for selecteer. This is with
    1 objective: TP - FP (both are mean percentages over all runs), 10 channels are used at once. Added cat_func with
    distance between electrodes on the grid as categorical distance function
    Better results but scoring seems incorrect, so unclear
                    optimize_mat_multiclicks4 and optimize_mat_multiclicks6
                    Optimizer2
2) GP sampler, no parallelization (inefficient), 1 objective: TP - FP, 4 channels are used as once
                    optimize_mat_multiclicks5 and optimize_mat_multiclicks7
                    Optimizer3
3) BruteForce sampler for completeness, 1 objective: TP - FP, 4 channels are used as once. Only ran for grasp
                    optimize_mat_multiclicks5
                    Optimizer4

TODO:
    - arrange different samplers into scenarios/parameters
    - keep multiple objective functions but use selecteer and grasp as a parameter, use only 1 matlab function
    - pass n_features as a parameter
    - specify a function for defining parameter space over channels
    - make a file optimization.py under ml, branch Optimizer into single or multi-objective or just keep single
    - keep track of individual TP and FP values somewhere
    - use feval instead of hardcoding matlab function name?

'''
import matlab.engine
import numpy as np
import sys
sys.path.insert(0, './src/')
sys.path.insert(0, '/home/julia/Documents/Python/PlumPy')
from pathlib import Path
from plumpy.utils.io import load_config

eng = matlab.engine.start_matlab()
s = eng.genpath('/home/julia/Documents/Python/PlumPy/plumpy/misc/matlab')
eng.addpath(s, nargout=0)

def optimize_mat_test(trial, n_features=6):
    function_name = 'opt_test'
    params = {
        'weights': np.array([trial.suggest_float(f'weights{i}', -10., 10., step=.1) for i in range(n_features)]),
    }
    prod, norm = eng.opt_test(params, nargout=2)
    #res = eng.feval(function_name, params)
    # [-.5, 3, .2, 9, .23, 1];
    print(prod, norm)
    return prod, norm

def optimize_mat_multiclicks(trial, n_features=4):
    function_name = 'opt_simulate_multiclicks'
    params = {
        'channels': np.array([trial.suggest_int(f'channel{i}', 97, 128, step=1) for i in range(n_features)]), # 33, 96
        'featureWeights': np.array([trial.suggest_float(f'featureWeight{i}', -1., 1, step=2) for i in range(n_features)])
    }
    tp, fp = eng.opt_simulate_multiclicks(params, nargout=2)
    print(tp, fp)
    return tp, fp

def optimize_mat_multiclicks2(trial, n_features=4):
    function_name = 'opt_simulate_multiclicks_grasp'
    channels = list(range(1, 129))
    channels.remove(121)
    params = {
        'channels': np.array([trial.suggest_categorical(f'channel{i}', channels) for i in range(n_features)]), # 33, 96
        'featureWeights': np.array([trial.suggest_float(f'featureWeight{i}', -1., 1, step=2) for i in range(n_features)]),
        'timeSmoothing': trial.suggest_float(f'timeSmoothing', 6., 22., step=4),
        'threshold': trial.suggest_float(f'threshold', .2, .9, step=.1),
        'activePeriod': trial.suggest_float(f'activePeriod', .5, 2, step=.5),
        'activeRate': trial.suggest_float(f'activeRate', .5, .9, step=.1),
    }
    tp, fp = eng.opt_simulate_multiclicks_grasp(params, nargout=2)
    print(tp, fp)
    return tp, fp

def optimize_mat_multiclicks3(trial, n_features=4):
    function_name = 'opt_simulate_multiclicks_selecteer'
    channels = list(range(1, 129))
    channels.remove(121)
    params = {
        'channels': np.array([trial.suggest_categorical(f'channel{i}', channels) for i in range(n_features)]), # 33, 96
        'featureWeights': np.array([trial.suggest_float(f'featureWeight{i}', -1., 1, step=2) for i in range(n_features)]),
        'timeSmoothing': trial.suggest_float(f'timeSmoothing', 6., 22., step=4),
        'threshold': trial.suggest_float(f'threshold', .2, .9, step=.1),
        'activePeriod': trial.suggest_float(f'activePeriod', .5, 2, step=.5),
        'activeRate': trial.suggest_float(f'activeRate', .5, .9, step=.1),
    }
    tp, fp = eng.opt_simulate_multiclicks_selecteer(params, nargout=2)
    print(tp, fp)
    return tp, fp

def optimize_mat_multiclicks4(trial, n_features=10):
    function_name = 'opt_simulate_multiclicks_grasp_mean'
    channels = list(range(33, 96)) # two upper grids
    params = {
        'channels': np.array([trial.suggest_categorical(f'channel{i}', channels) for i in range(n_features)]), # 33, 96
        'featureWeights': np.array([trial.suggest_float(f'featureWeight{i}', -1., 1, step=0.2) for i in range(n_features)]),
        'timeSmoothing': trial.suggest_float(f'timeSmoothing', 6., 22., step=4),
        'threshold': trial.suggest_float(f'threshold', .2, .9, step=.1),
        'activePeriod': trial.suggest_float(f'activePeriod', .5, 2, step=.5),
        'activeRate': trial.suggest_float(f'activeRate', .5, .9, step=.1),
    }
    tp, fp = eng.opt_simulate_multiclicks_grasp_mean(params, nargout=2)
    print(tp, fp)
    return tp - fp

def optimize_mat_multiclicks5(trial, n_features=4):
    function_name = 'opt_simulate_multiclicks_grasp_mean'
    channels = [63, 61, 64, 59, 60, 62, 57, 58, 79, 77, 75, 73, 71, 69, 67, 65]
    params = {
        'channels': np.array([trial.suggest_categorical(f'channel{i}', channels) for i in range(n_features)]), # 33, 96
        'featureWeights': np.array([trial.suggest_float(f'featureWeight{i}', 0, 1, step=0.2) for i in range(n_features)]),
        'timeSmoothing': trial.suggest_float(f'timeSmoothing', 6., 34., step=4),
        'threshold': trial.suggest_float(f'threshold', .2, .9, step=.1),
        'activePeriod': trial.suggest_float(f'activePeriod', .5, 2, step=.5),
        'activeRate': trial.suggest_float(f'activeRate', .5, .9, step=.1),
    }
    tp, fp = eng.opt_simulate_multiclicks_grasp_mean(params, nargout=2)
    print(tp, fp)
    return tp - fp

def optimize_mat_multiclicks6(trial, n_features=10):
    channels = list(range(1, 33)) + list(range(97, 129)) # two bottom grids
    channels.remove(121)
    params = {
        'channels': np.array([trial.suggest_categorical(f'channel{i}', channels) for i in range(n_features)]), # 33, 96
        'featureWeights': np.array([trial.suggest_float(f'featureWeight{i}', -1., 1, step=0.2) for i in range(n_features)]),
        'timeSmoothing': trial.suggest_float(f'timeSmoothing', 6., 22., step=4),
        'threshold': trial.suggest_float(f'threshold', .2, .9, step=.1),
        'activePeriod': trial.suggest_float(f'activePeriod', .5, 2, step=.5),
        'activeRate': trial.suggest_float(f'activeRate', .5, .9, step=.1),
    }
    tp, fp = eng.opt_simulate_multiclicks_selecteer(params, nargout=2)
    print(tp, fp)
    return tp - fp

def optimize_mat_multiclicks7(trial, n_features=4):
    channels = [104, 103, 102, 101, 100, 98, 99, 97, 113, 111, 110, 109, 108, 107, 106, 105]
    params = {
        'channels': np.array([trial.suggest_categorical(f'channel{i}', channels) for i in range(n_features)]), # 33, 96
        'featureWeights': np.array([trial.suggest_float(f'featureWeight{i}', 0, 1, step=0.2) for i in range(n_features)]),
        'timeSmoothing': trial.suggest_float(f'timeSmoothing', 6., 34., step=4),
        'threshold': trial.suggest_float(f'threshold', .2, .9, step=.1),
        'activePeriod': trial.suggest_float(f'activePeriod', .5, 2, step=.5),
        'activeRate': trial.suggest_float(f'activeRate', .5, .9, step=.1),
    }
    tp, fp = eng.opt_simulate_multiclicks_selecteer(params, nargout=2)
    print(tp, fp)
    return tp - fp

def optimize_mat_multiclicks8(trial, task, n_features):
    hdr = {}
    task_, brainFunction = task.split('_')
    hdr['session'] = np.array([17., 18.])
    if brainFunction == 'grasp':
        channels = [63, 61, 64, 59, 60, 62, 57, 58, 79, 77, 75, 73, 71, 69, 67, 65]
        hdr['sequenceDuration'] = 1.
        hdr['brainFunction'] = 'Grasp'
    elif brainFunction == 'selecteer':
        channels = [104, 103, 102, 101, 100, 98, 99, 97, 113, 111, 110, 109, 108, 107, 106, 105]
        hdr['sequenceDuration'] = 3.
        hdr['brainFunction'] = 'Selecteer'
    else:
        raise NotImplementedError
    params = {
        'channels': np.array([trial.suggest_categorical(f'channel{i}', channels) for i in range(n_features)]), # 33, 96
        'featureWeights': np.array([trial.suggest_float(f'featureWeight{i}', 0.5, 1, step=0.5) for i in range(n_features)]),
        'lowFreq': trial.suggest_float(f'lowFreq', 55., 95., step=10),
        'timeSmoothing': trial.suggest_float(f'timeSmoothing', 0., 12., step=4),
        'threshold': trial.suggest_float(f'threshold', .1, .9, step=.1),
        'activePeriod': trial.suggest_float(f'activePeriod', .2, 2, step=.2),
        'activeRate': trial.suggest_float(f'activeRate', .3, .9, step=.1),
    }
    params['highFreq'] = trial.suggest_float(f'highFreq', params['lowFreq'], 145., step=10)
    tp, fp = eng.opt_simulate_multiclicks(hdr, params, nargout=2)
    print(tp, fp)
    return tp - fp




##
cfg = load_config('/Fridge/users/julia/project_corticom/cc2/config_opt_matlab.yml')
# from plumpy.ml.Optimizer import Optimizer
# opt = Optimizer(cfg)
# if cfg['task'] == 'multiclicks_grasp':
#     opt.optimize(obj_fun=optimize_mat_multiclicks2, directions=['maximize', 'minimize'])
# elif cfg['task'] == 'multiclicks_selecteer':
#     opt.optimize(obj_fun=optimize_mat_multiclicks3, directions=['maximize', 'minimize'])
# self.study.best_trials

# from plumpy.ml.Optimizer import Optimizer2
# opt = Optimizer2(cfg)
# if cfg['task'] == 'multiclicks_grasp':
#     opt.optimize(obj_fun=optimize_mat_multiclicks4,
#                  direction='maximize',
#                  n_features=10,
#                  cat_dist_fun=channel_dist)

# from plumpy.ml.Optimizer import Optimizer3
# opt = Optimizer3(cfg)
# if cfg['task'] == 'multiclicks_grasp':
#     opt.optimize(obj_fun=optimize_mat_multiclicks5,
#                  direction='maximize')

# from plumpy.ml.Optimizer import Optimizer4
# opt = Optimizer4(cfg)
# if cfg['task'] == 'multiclicks_grasp':
#     opt.optimize(obj_fun=optimize_mat_multiclicks5,
#                  direction='maximize')

# from plumpy.ml.Optimizer import Optimizer2
# opt = Optimizer2(cfg)
# if cfg['task'] == 'multiclicks_selecteer':
#     opt.optimize(obj_fun=optimize_mat_multiclicks6,
#                  direction='maximize',
#                  n_features=10,
#                  cat_dist_fun=channel_dist)

# from plumpy.ml.Optimizer import Optimizer3
# opt = Optimizer3(cfg)
# if cfg['task'] == 'multiclicks_selecteer':
#     opt.optimize(obj_fun=optimize_mat_multiclicks7,
#                  direction='maximize')


params = load_config('/Fridge/users/julia/project_corticom/cc2/multiclicks_grasp_paramsets.yml')
from plumpy.ml.Optimizer import Optimizer5
opt = Optimizer5(cfg, obj_fun=lambda trial: optimize_mat_multiclicks8(trial, task=cfg['task'], n_features=cfg['n_features']),
                      direction='maximize',
                      type_sampler=cfg['model_type'].split('_')[0],
                      n_features=cfg['n_features'])
if not opt.study.trials: # if empty study
    for k, v in params.items():
        opt.enquire(v)
opt.optimize()


##
