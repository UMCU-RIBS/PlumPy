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

pd.set_option('display.max_rows', 500)


def rsquare_one(data, events, config, run=None, plot=True):
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
    name = config["subject"]
    task_info = load_config(f'{config["meta_path"]}/task_{task}.yml')
    subj_cfg = load_config(f'{config["meta_path"]}/{name}/{name}.yml')
    plot_path = config["plot_path"]
    sr_post = config['preprocess']['target_sampling_rate']
    grid = load_grid(subj_cfg['grid_map'])

    al_vals = {k: None for k in params['duration']}
    task_events = events['events'].values
    task_times = events[f'samples_{sr_post}Hz'].values
    t_events_sec = events['times_sec'].values

    # task info
    all_dict = task_info['codes'][variant]
    rest_cues = {k:v for k, v in all_dict.items() if v == 'rust'}
    active_cues = {k:v for k, v in all_dict.items() if v != 'rust'}

    # smooth data
    from plumpy.sigproc.general import smooth_signal_1d
    x = smooth_signal_1d(data['hfb'], n=params['n_smooth'])

    # rsquare
    dur = task_info['duration_rest']
    rsquare = {k:[] for k in active_cues.values()}
    mr = []
    for r in rest_cues.keys():
        if r in task_events:
            for ir in np.where(task_events == r)[0]:
                start_go = ir + 1
                mr.append(np.mean(x[task_times[start_go]:task_times[start_go] + dur * sr_post], 0))
    mr = np.array(mr)

    for w, word in active_cues.items():
        mw = []
        if w in task_events:
            for iw in np.where(task_events == w)[0]:
                if run in config['dyn_runs']:
                    start_go = iw + 2 # cue 31 31 cue
                else:
                    start_go = iw + 1 # cue 31 cue
                mw.append(np.mean(x[task_times[start_go]:task_times[start_go] + dur * sr_post], 0))
        mw = np.array(mw)

        for c in range(mr.shape[-1]):
            pears = np.corrcoef(np.vstack([np.zeros_like(mr), np.ones_like(mw)])[..., c], np.vstack([mr, mw])[..., c])[0,1]
            rsquare[word].append(np.sign(pears) * pears ** 2)
        rsquare[word] = np.array(rsquare[word])

        if plot:
            plot_on_grid(grid, rsquare[word], label=f'{word}-rest', colormap='vlag', xmin=-1, xmax=1)
            save_plot(plot_path, name=f'{task}_{run}_rsquare_{word}')
    return rsquare

##
def rsquare_mean(config):
    task = config['task']
    name = config["subject"]
    subj_cfg = load_config(f'{config["meta_path"]}/{name}/{name}.yml')
    plot_path = config["plot_path"]
    grid = load_grid(subj_cfg['grid_map'])
    task_info = load_config(f'{config["meta_path"]}/task_{task}.yml')
    out = { k:[] for k in list(task_info['codes']['1-7'].values())[:-1] + list(task_info['codes']['8-14'].values())[:-1] }

    for run in config['include_runs']:
        print(run)
        data, events = load_processed(task, run, config)
        temp = rsquare_one(data, events, config, run, plot=False)
        for k, v in temp.items():
            out[k].append(v)

    for k, v in out.items():
        out[k] = np.mean(np.array(out[k]), 0)
        plot_on_grid(grid, out[k], label=f'{k}-rest', colormap='vlag', xmin=-1, xmax=1)
        save_plot(plot_path, name=f'{task}_mean_rsquare_{k}_{len(config["include_runs"])}runs')
    return out

