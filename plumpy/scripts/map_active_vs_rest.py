'''
python src/map_active_vs_rest.py -c /Fridge/users/julia/project_corticom/cc2/config_14nav.yml
'''
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


def map_active_one(data, events, config, run=None, plot=True):
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
    rest_prep = [k for k, v in all_dict.items() if v == 'rust']
    active_prep = [k for k, v in all_dict.items() if v != 'rust']

    # smooth data
    from plumpy.sigproc.general import smooth_signal_1d
    x = smooth_signal_1d(data['hfb'], n=params['n_smooth'])

    # compare to a baseline
    if params['comparison'] == 't-baseline':
        rest_data, rest_evs = load_processed(task, run, config)
        rest_events = rest_evs['events']
        rest_times = rest_evs['samples_100Hz']
        y = smooth_signal_1d(rest_data['hfb'], n=params['n_smooth'])
        if 0 not in rest_events and 1 not in rest_events:  # s009 of 14nav
            st_rest = rest_times[rest_events == 10][0]
            en_rest = rest_times[rest_events == 50][0]
        else:
            st_rest = rest_times[rest_events == 0][0]
            en_rest = rest_times[rest_events == 1][0]
        xz, scaler = zscore(x, y=y, xmin=st_rest, xmax=en_rest, units='samples')
        yz, _ = zscore(y, xmin=st_rest, xmax=en_rest, units='samples')
        prng2 = RandomState(599)
        ind_rest = np.sort(prng2.randint(st_rest, en_rest, 28))
        #for dur in [2 * sr_post, 4 * sr_post, 6 * sr_post]:
        for dur in params['duration']:
            tw = []
            for w in active_prep:
                if w in task_events:
                    for iw in np.where(task_events == w)[0]:
                        if run in config['dyn_runs']:
                            start_go = iw + 2 # cue 31 31 cue
                        else:
                            start_go = iw + 1 # cue 31 cue
                        tw.append(np.mean(xz[task_times[start_go]:task_times[start_go] + dur*sr_post], 0))
            tw = np.array(tw)
            tr = []
            for r in ind_rest:
                tr.append(np.mean(yz[r:r + dur*sr_post], 0))
            tr = np.array(tr)
            a = [ttest_rel(tw[:, i], tr[:, i]) for i in range(xz.shape[-1])]
            ts = np.array([i.statistic for i in a])
            al_vals[dur] = ts
    else:
        if params['comparison'] == 'z-start':
            x_, scaler2 = zscore(x, xmin=t_events_sec[task_events == 50][0], duration=5, units='seconds', sr=sr_post)
        elif params['comparison'] == 'z-rest':
            temp = []
            for c, ir in enumerate(task_events):
                if ir in rest_prep:
                    temp.append(x[task_times[c + 1]:task_times[c + 1] + task_info['duration_rest'] * sr_post])
            x_, scaler2 = zscore(x, np.concatenate(temp), xmin=0, duration=len(temp)*task_info['duration_rest']*sr_post, units='samples')
        else:
            raise NotImplementedError

        #for dur in [2 * sr_post, 3 * sr_post, 4 * sr_post, 6 * sr_post]:
        for dur in params['duration']:
            zw = []
            for w in active_prep:
                if w in task_events:
                    for iw in np.where(task_events == w)[0]:
                        if run in config['dyn_runs']:
                            start_go = iw + 2 # cue 31 31 cue
                        else:
                            start_go = iw + 1 # cue 31 cue
                        zw.append(np.mean(x_[task_times[start_go]:task_times[start_go] + dur * sr_post], 0))
            zw = np.mean(np.array(zw), 0)
            al_vals[dur] = zw

        if plot:
            for dur in al_vals.keys():
                if params['comparison'] == 't_baseline':
                    plot_on_grid(grid, al_vals[dur], label=f'Words-rest ({dur} s)', colormap='vlag', xmin=-10, xmax=10)
                else:
                    plot_on_grid(grid, al_vals[dur], label=f'Words-rest ({dur} s)', colormap='vlag', xmin=-2, xmax=2)
                save_plot(plot_path, name=f'{task}_{run}_{params["comparison"]}_words_dur{dur}s')
        return al_vals

##
def map_active_mean(config):
    params = config['map_activity']
    task = config['task']
    name = config["subject"]
    subj_cfg = load_config(f'{config["meta_path"]}/{name}/{name}.yml')
    plot_path = config["plot_path"]
    grid = load_grid(subj_cfg['grid_map'])
    out = []

    for run in config['include_runs']:
        print(run)
        data, events = load_processed(task, run, config)
        out.append(map_active_one(data, events))
        if params['comparison'] == 't_baseline':
            plot_on_grid(grid, np.mean(np.array(out), 0), label=f'Words-rest (6 s)', colormap='vlag', xmin=-10,
                         xmax=10)
        else:
            plot_on_grid(grid, np.mean(np.array(out), 0), label=f'Words-rest (6 s)', colormap='vlag', xmin=-2.5,
                         xmax=2.5)
        save_plot(plot_path,
                  name=f'{task}_mean_{params["comparison"]}_words_dur6s_{len(subj_cfg["include_runs"])}runs')
    return out
