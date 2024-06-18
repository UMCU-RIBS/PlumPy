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
featureWeights combine information across channels resulting in n signals, where n is the number of frequency bins
linearWeights combine information across frequency bins resulting in 1 signal used for calculating clicks

TODO:
    - add running over all sessions
    - add which parameters need to be optimized and which are set


'''
import warnings

import matlab.engine
import numpy as np
import optuna
import sys
sys.path.insert(0, './src/')
sys.path.insert(0, '/home/julia/Documents/Python/PlumPy')
from plumpy.utils.io import load_config
from plumpy.utils.general import to_list

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
    if isinstance(selection, str):
        if selection == 'top2grids':
            channels = grid[:8]
        elif selection == 'bottom2grids':
            channels = grid[-8:]
        elif selection == 'hand16':
            channels = grid[3:5]
        elif selection == 'speech16':
            channels = grid[-4:-2]
        elif selection == 'all':
            channels = grid.copy()
        else:
            raise NotImplementedError
        channels = [int(i) for i in channels.flatten()]

    elif isinstance(selection, int) or isinstance(selection, list):
        channels = to_list(selection)
    else:
        raise NotImplementedError

    # remove bad channel 121
    if 121 in channels:
        warnings.warn('Excluding bad channel 121. Make sure this is reflected in n_features')
        channels.remove(121)
    return channels

def optimize_mat_multiclicks(trial, subject, task, brain_function, sequence_duration, session, n_features, n_bins=2, draw_channels_from='all', id=None):
    # populate hdr with brainFunction and session values,
    hdr = {}
    hdr['subject'] = subject.upper()
    hdr['task'] = task[:1].capitalize() + task[1:]
    hdr['session'] = np.array(session)
    hdr['sequenceDuration'] = sequence_duration
    hdr['brainFunction'] = brain_function.capitalize()
    print(f'Running optimization: {id}')
    print(f'Number of features: {n_features}')
    print(f'Draw from channels: {draw_channels_from}')
    if brain_function == 'grasp' and sequence_duration not in [1, 2, 3, 5] \
        or brain_function == 'selecteer' and sequence_duration not in [3, 4, 6, 8]:
            warnings.warn(f'{brain_function.capitalize()} and {sequence_duration} may not be compatible')

    # specify which channels to draw from
    channels = channel_selector(draw_channels_from)

    # set reasonable bounds for certain parameters
    lowFreqBins = {0:[5., 15.], 1:[55., 95.]}
    highFreqBins = {0: [15., 35.], 1: [95., 145.]}
    linWeights = {0: [-1., 0.], 1: [.5, 1]}

    # initialize parameters from specified distributions
    params = {
        'featureWeights': np.array([trial.suggest_float(f'featureWeight{i}', 0., 1, step=.2)
                                                                        for i in range(n_features)]),
        'linearWeights': np.array([trial.suggest_float(f'linearWeights{i}', linWeights[i][0],
                                                                            linWeights[i][1], step=.5)
                                                                        for i in range(n_bins)]),
        'timeSmoothing': trial.suggest_float(f'timeSmoothing', 2., 30., step=4),
        'threshold': trial.suggest_float(f'threshold', .1, 2., step=.1),
        'activePeriod': trial.suggest_float(f'activePeriod', .5, 1.5, step=.2),
        'activeRate': trial.suggest_float(f'activeRate', .5, 1., step=.1),
        'lowFreq': np.array([trial.suggest_float(f'lowFreq{i}', lowFreqBins[i][0], lowFreqBins[i][1], step=5)
                                                                        for i in range(n_bins)]),
    }
    # set highFreq dynamically depending on the lowFreq value, allow for highFreq - lowFreq to be at least 10
    params['highFreq'] = np.array([trial.suggest_float(f'highFreq{i}', params[f'lowFreq'][i]+10, highFreqBins[i][1], step=5) for i in range(n_bins)])

    # keep all channels if the size of the drawing pool equals the number of features
    if len(channels) == n_features:
        params['channels'] = np.array(channels)
    else:
        params['channels'] = np.array([trial.suggest_categorical(f'channel{i}', channels) for i in range(n_features)])

    # prune trials with duplicate channels selected, e.g. [64, 64, 59, 77]
    if len(set(params['channels'])) < len(params['channels']):
        raise optuna.exceptions.TrialPruned()

    tp, fp = eng.opt_simulate_multiclicks(hdr, params, nargout=2)

    # keep track of true positive and false positive components of the objective
    trial.set_user_attr("truePositive", tp)
    trial.set_user_attr("falsePositive", fp)
    print(tp, fp)
    accuracy = (tp/100 + 1 - fp/100) / 2
    f1 = 2 * tp/100 / (2 * tp/100 + fp/100 + (1-tp/100))
    specificity = (1-fp/100) / (1-tp/100 + fp/100)
    sensitivity = tp/100 / (tp/100 + (1-tp/100))
    #return tp - fp
    return accuracy


##
# load configuration file with parameters for this study
cfg = load_config('/Fridge/users/julia/project_corticom/cc2/config_opt_matlab.yml')

# initialize the optimizer
from plumpy.ml.optimization import Optimizer
id = f'sub-{cfg["subject"]}_task-{cfg["task"]}_fun-{cfg["brain_function"]}_' \
                           f'dur-{str(cfg["sequence_duration"])}_feat-{str(cfg["n_features"])}_' \
                           f'chan-{cfg["channels_from"]}_bins-{str(cfg["n_bins"])}_opt-{cfg["sampler"]}'

opt = Optimizer(id, cfg, obj_fun=lambda trial: optimize_mat_multiclicks(trial,
                                                                    subject=cfg['subject'],
                                                                    task=cfg['task'],
                                                                    brain_function=cfg['brain_function'],
                                                                    sequence_duration=cfg['sequence_duration'],
                                                                    session=cfg['session'],
                                                                    n_features=cfg['n_features'],
                                                                    n_bins=cfg['n_bins'],
                                                                    draw_channels_from=cfg['channels_from'],
                                                                    id=cfg['sampler']),
                      direction='maximize',
                      n_features=cfg['n_features'])

# add manual parameter sets here
if cfg['add_external_sets']:
    # load manually defined parameter sets to add to the optimization, could provide a good starting point
    params = load_config(
        f'/Fridge/users/julia/project_corticom/cc2/{cfg["task"]}_{cfg["brain_function"]}_paramsets.yml')
    if not opt.study.trials: # if empty study
        for k, v in params.items():
            opt.enquire(v)

# run optimization
opt.optimize()

