'''
python src/map_active_vs_rest.py -c /Fridge/users/julia/project_corticom/cc2/config_14nav.yml
'''
import numpy as np
import pandas as pd
import sys
from scipy.stats import ttest_rel
from numpy.random import RandomState
sys.path.insert(0, './src/')
sys.path.insert(0, '/home/julia/Documents/Python/PlumPy')
from plumpy.utils.io import load_config, load_grid, load_processed
from plumpy.utils.plots import *
from plumpy.ml.general import zscore
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)


def active_trace_one(data, events, config, run=None, plot=True):
    '''

    :param data:
    :param events:
    :param config:
    :param run: neede for loading the corresponding rest
    :return:
    '''
    params = config['map_activity']
    task = config['task']
    variant = config['order'][run]
    task_info = load_config(f'{config["meta_path"]}/task_{task}.yml')
    plot_path = str(Path(config["plot_path"])/'active_traces')
    sr_post = config['preprocess']['target_sampling_rate']
    features = list(config['preprocess']['bands'].keys())
    task_events = events['events'].values
    task_times = events[f'samples_{sr_post}Hz'].values

    # task info
    all_dict = task_info['codes'][variant]
    rest_cues = {k:v for k, v in all_dict.items() if v == 'rust'}
    active_cues = {k:v for k, v in all_dict.items() if v != 'rust'}

    # smooth data
    from plumpy.sigproc.general import smooth_signal_1d
    x = smooth_signal_1d(data['hfb'], n=params['n_smooth'])

    # z-score
    temp = []
    for c, ir in enumerate(task_events):
        if ir in rest_cues.keys():
            temp.append(x[task_times[c + 1]:task_times[c + 1] + task_info['duration_rest'] * sr_post])
    x_, scaler2 = zscore(x, np.concatenate(temp), xmin=0, duration=len(temp) * task_info['duration_rest'] * sr_post,
                         units='samples')

    # plot traces
    tmin = params['epochs']['tmin']
    tmax = params['epochs']['tmax']
    dur = tmax - tmin
    traces = {k: [] for k in active_cues.values()}
    for w, word in active_cues.items():
        mw = []
        if w in task_events:
            for iw in np.where(task_events == w)[0]:
                if run in config['dyn_runs']:
                    start_go = iw + 2 # cue 31 31 cue
                else:
                    start_go = iw + 1 # cue 31 cue
                start_tmin = int(round(task_times[start_go] + tmin * sr_post))
                start_tmax = int(round(task_times[start_go] + tmin * sr_post + dur * sr_post))
                mw.append(x_[start_tmin:start_tmax])
        mw = np.array(mw)
        nrep, ntime, nch = mw.shape

        if plot:
            for ch in range(nch):
                plt.plot(mw[..., ch].T, alpha=.8, color='black', linewidth=.3)
                plt.plot(np.mean(mw[..., ch], 0), linewidth=2, color='red')
                plt.xticks(range(0, ntime+1, sr_post), np.arange(tmin, tmax + 1))
                plt.title(f'{word}, ch {ch+1}')
                save_plot(plot_path, name=f'{task}_{run}_{"".join(features)}_trace_{word}_ch{ch+1}')
        traces[word] = mw
    return traces

##
def active_trace_mean(config):
    task = config['task']
    plot_path = str(Path(config["plot_path"])/'active_traces')
    task_info = load_config(f'{config["meta_path"]}/task_{task}.yml')
    sr_post = config['preprocess']['target_sampling_rate']
    params = config['map_activity']
    tmin = params['epochs']['tmin']
    tmax = params['epochs']['tmax']
    out = { k:[] for k in list(task_info['codes']['1-7'].values())[:-1] + list(task_info['codes']['8-14'].values())[:-1] }
    channels = [i-1 for i in params['channels']]
    features = list(config['preprocess']['bands'].keys())

    for run in config['include_runs']:
        print(run)
        data, events = load_processed(task, run, config)
        temp = active_trace_one(data, events, config, run, plot=False)
        for k, v in temp.items():
            out[k].append(v)

    for k, v in out.items():
        v = np.array(v) # runs x repetitions x time x channels
        nruns, nrep, ntime, nchan = v.shape
        for ch in channels:
            plt.figure()
            d = v[..., ch].reshape(-1, ntime)
            plt.plot(d.T, alpha=.3, color='black', linewidth=.2)
            plt.plot(np.mean(d, 0), linewidth=2, color='red')
            plt.xticks(range(0, ntime + 1, sr_post), np.arange(tmin, tmax + 1))
            plt.title(f'{k}, ch {ch+1}')
            save_plot(plot_path, name=f'{task}_mean_{"".join(features)}_trace_{k}_{nruns}runs_ch{ch+1}')
    return out

