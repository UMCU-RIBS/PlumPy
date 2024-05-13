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


Different scenarios I have tried:
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

Above uses the old definition of TP (can be >100) and FP (counts)

**** after refactoring code and adjusting definition of TP and FP (<=100 and percentage both) ****
4) GP sampler, no parallelization, 1 objective: TP - FP, 4 channels are used as once, also optimize
    frequency: lowFreq and highFreq. 700 trials Selecteer and Grasp. Trials with duplicate channel selections are pruned.
    Grasp (top 2 grids) and selecteer (bottom 2 grids).
                    optimize_mat_multiclicks
                    Optimizer

TODO:
    - handle varying parameter spaces in one objective function: e.g. weights 0 to 1 or only 1s

'''
import matlab.engine
import numpy as np
import optuna
import sys
sys.path.insert(0, './src/')
sys.path.insert(0, '/home/julia/Documents/Python/PlumPy')
from plumpy.utils.io import load_config

eng = matlab.engine.start_matlab()
s = eng.genpath('/home/julia/Documents/Python/PlumPy/plumpy/misc/matlab')
eng.addpath(s, nargout=0)

def channel_selector(selection:str='all'):
    """
    Select channels to draw from during the optimization procedure
    Args:
        selection (str): from top2, bottom2. grasp16, selecteer16 and all

    Returns:
        list: channels
    """
    from plumpy.utils.io import load_grid
    grid = load_grid(
        '/Fridge/bci/data/23-171_CortiCom/E_ResearchData/UBCI-CC02/7_Electrode_localization/Electrode order map/CortecGrid_electrode_label.txt')
    if selection == 'top2grids':
        channels = grid[:8]
    elif selection == 'bottom2grids':
        channels = grid[-8:]
    elif selection == 'grasp16':
        channels = grid[3:5]
    elif selection == 'selecteer16':
        channels = grid[-4:-2]
    elif selection == 'all':
        channels = grid.copy()
    else:
        raise NotImplementedError
    channels = [int(i) for i in channels.flatten()]

    # remove bad channel 121
    if 121 in channels:
        channels.remove(121)
    return channels

def optimize_mat_multiclicks(trial, task, n_features, draw_channels_from='all', id=None):
    # populate hdr with brainFunction and session values,
    # specify which channels to draw from
    hdr = {}
    task_, brainFunction = task.split('_')
    hdr['session'] = np.array([17., 18.])
    if brainFunction == 'grasp':
        hdr['sequenceDuration'] = 1.
        hdr['brainFunction'] = 'Grasp'
    elif brainFunction == 'selecteer':
        hdr['sequenceDuration'] = 3.
        hdr['brainFunction'] = 'Selecteer'
    else:
        raise NotImplementedError
    channels = channel_selector(draw_channels_from)
    print(f'Running optimization: {id}')
    print(f'Number of features: {n_features}')
    print(f'Draw from channels: {draw_channels_from}')

    # initialize parameters from specified distributions
    params = {
        'channels': np.array([trial.suggest_categorical(f'channel{i}', channels) for i in range(n_features)]), # 33, 96
        'featureWeights': np.array([trial.suggest_float(f'featureWeight{i}', 0., 1, step=0.2) for i in range(n_features)]),
        #'featureWeights': np.array([trial.suggest_float(f'featureWeight{i}', 1, 1, step=1) for i in range(n_features)]),
        'timeSmoothing': trial.suggest_float(f'timeSmoothing', 2., 14., step=4),
        'threshold': trial.suggest_float(f'threshold', .1, .9, step=.1),
        'activePeriod': trial.suggest_float(f'activePeriod', .2, 2, step=.2),
        'activeRate': trial.suggest_float(f'activeRate', .3, .9, step=.1),
        'lowFreq': trial.suggest_float(f'lowFreq', 55., 95., step=10),
    }
    # set highFreq dynamically depending on the lowFreq value, allow for highFreq - lowFreq to be at least 10
    params['highFreq'] = trial.suggest_float(f'highFreq', params['lowFreq']+10, 145., step=10)

    # prune trials with duplicate channels selected, e.g. [64, 64, 59, 77]
    if len(set(params['channels'])) < len(params['channels']):
        raise optuna.exceptions.TrialPruned()

    tp, fp = eng.opt_simulate_multiclicks(hdr, params, nargout=2)

    # keep track of true positive and false positive components of the objective
    trial.set_user_attr("truePositive", tp)
    trial.set_user_attr("falsePositive", fp)
    print(tp, fp)
    return tp - fp


##
# load configuration file with parameters for this study
cfg = load_config('/Fridge/users/julia/project_corticom/cc2/config_opt_matlab.yml')

# load manually defined parameter sets to add to the optimization, could provide a good starting point
params = load_config(f'/Fridge/users/julia/project_corticom/cc2/multiclicks_{cfg["task"].split("_")[-1]}_paramsets.yml')

# initialize the optimizer
from plumpy.ml.optimization import Optimizer
opt = Optimizer(cfg, obj_fun=lambda trial: optimize_mat_multiclicks(trial,
                                                                    task=cfg['task'],
                                                                    n_features=cfg['n_features'],
                                                                    draw_channels_from=cfg['channels_from'],
                                                                    id=cfg['model_type']),
                      direction='maximize',
                      type_sampler=cfg['model_type'].split('_')[0],
                      n_features=cfg['n_features'])

# add manual parameter sets here
if cfg['add_external_sets']:
    if not opt.study.trials: # if empty study
        for k, v in params.items():
            opt.enquire(v)

# run optimization
opt.optimize()

