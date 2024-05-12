'''
virtual envirornment with the matlab engine: matlab (miniconda)
had to be a separate env because current matlab version is 23.2.1, it is only compatible with python < 3.12
so I ran:

conda create -n matlab python=3.10
python -m pip install matlabengine==23.2.1

prior to this, followed instructions here: https://pypi.org/project/matlabengine/

python /home/julia/Documents/Python/PlumPy/plumpy/scripts/optimize_params.py

'''
import matlab.engine
import numpy as np
import sys
sys.path.insert(0, './src/')
sys.path.insert(0, '/home/julia/Documents/Python/PlumPy')
from plumpy.utils.io import load_config

eng = matlab.engine.start_matlab()
s = eng.genpath('/home/julia/Documents/Python/PlumPy/plumpy/misc/matlab')
eng.addpath(s, nargout=0)
# data = [-.5, 3, .2, 9, .23, 1];

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

def channel_dist(ch1, ch2):
    import math
    from plumpy.utils.io import load_grid
    grid = load_grid(
        '/Fridge/bci/data/23-171_CortiCom/E_ResearchData/UBCI-CC02/7_Electrode_localization/Electrode order map/CortecGrid_electrode_label.txt')

    sqz = lambda x: np.array(x).squeeze()
    pos = lambda x: np.where(grid==x)
    p1 = sqz(pos(ch1))
    p2 = sqz(pos(ch2))
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])



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

from plumpy.ml.Optimizer import Optimizer3
opt = Optimizer3(cfg)
if cfg['task'] == 'multiclicks_selecteer':
    opt.optimize(obj_fun=optimize_mat_multiclicks7,
                 direction='maximize')

