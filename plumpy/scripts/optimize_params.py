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
from plumpy.ml.Optimizer import Optimizer
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
        'channels': np.array([trial.suggest_int(f'channel{i}', 33, 96, step=1) for i in range(n_features)]),
        'featureWeights': np.array([trial.suggest_float(f'featureWeight{i}', -1., 1, step=.2) for i in range(n_features)])
    }
    tp, fp = eng.opt_simulate_multiclicks(params, nargout=2)
    #res = eng.feval(function_name, params)
    # [-.5, 3, .2, 9, .23, 1];
    print(tp, fp)
    return tp, fp

##
cgf = load_config('/Fridge/users/julia/project_corticom/cc2/config_opt_matlab.yml')
opt = Optimizer(cgf)
opt.optimize(obj_fun=optimize_mat_multiclicks, directions=['maximize', 'minimize'])
# self.study.best_trials