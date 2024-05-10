'''
virtual envirornment with the matlab engine: matlab (miniconda)
had to be a separate env because current matlab version is 23.2.1, it is only compatible with python < 3.12
so I ran:

conda create -n matlab python=3.10
python -m pip install matlabengine==23.2.1

prior to this, followed instructions here: https://pypi.org/project/matlabengine/

'''
import matlab.engine
import numpy as np
from Optimizer import Optimizer
from plumpy.utils.io import load_config

eng = matlab.engine.start_matlab()
s = eng.genpath('/home/julia/Documents/Python/PlumPy/plumpy/misc/matlab')
eng.addpath(s, nargout=0)
function_name = 'opt_test'
# data = [-.5, 3, .2, 9, .23, 1];

def optimize_mat_test(trial, n_features=6):
    params = {
        'weights': np.array([trial.suggest_float(f'weights{i}', -10., 10., step=.1) for i in range(n_features)]),
    }
    res = eng.test_fun(params)
    #res = eng.feval(function_name, 'params')
    print(res)
    return res


##
cgf = load_config('/Fridge/users/julia/project_corticom/cc2/config_opt_matlab.yml')
opt = Optimizer(cgf)
opt.optimize(obj_fun=optimize_mat_test)