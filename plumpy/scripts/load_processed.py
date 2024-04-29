'''
'''
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, './src/')
sys.path.insert(0, '/home/julia/Documents/Python/PlumPy')
from plumpy.utils.io import load_config, load_blackrock
from plumpy.utils.plots import *
from plumpy.sigproc.general import calculate_rms
from plumpy.sigproc.process import process_mne
pd.set_option('display.max_rows', 500)


##

def run_dqc(subj_cfg, task, run, preload=False):
    ## set params
    tag = f'{task}_{run}'
    subject = load_config(subj_cfg)
    cfg = load_config(subject['tasks'][task])
    sr_post = cfg['target_sampling_rate']
    assert run in cfg['all_runs'], 'No such run'

    ## load data
    data, events, units = load_blackrock(cfg['raw_paths'][run])
    sr_raw = data['samp_per_s']
    plot_data(data, events, units=units)
    proc_path = subject['data_path']
    plot_path = subject['plot_path']
    save_plot(plot_path, name=tag + '_raw_triggers')

    ##
    seg_id = 0
    t_data = data["data_headers"][seg_id]["Timestamp"] / data["samp_per_s"]
    t_events = np.array(events['digital_events']['TimeStamps']) / data["samp_per_s"]
    c_events = np.array(events['digital_events']['UnparsedData'])

    ## find sample_ids that correspond to markers
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    event_samples, event_sample_ids = zip(*[find_nearest(t_data, i) for i in t_events])
    # verify
    # plt.figure()
    # plt.plot(data["data"][seg_id][5])
    # plt.vlines(x=event_sample_ids, ymin=-400, ymax=400, color='black')

    ## channels
    ch_name = subject['grid_map']
    channels = pd.read_csv(ch_name, header=None)
    channels = [int(i.strip('ch')) for i in channels[0]]
    grid = np.array(channels).reshape(-1, 8)

    ## plot rms
    rms = calculate_rms(data["data"][seg_id])
    plot_on_grid(grid, rms, label='RMS', colormap='viridis_r')
    save_plot(plot_path, name=tag + '_raw_rms')

    ## plot means
    means = np.mean(data["data"][seg_id], 1)
    plot_on_grid(grid, means, label='Mean', colormap='viridis_r')
    save_plot(plot_path, name=tag + '_raw_mean')

    ## plot channels on a grid
    stds = np.std(data["data"][seg_id], 1)
    outliers = np.where(stds > np.percentile(stds, 95))[0]
    #outliers = np.where(stds > 200)[0]
    if not preload:
        plot_signals_on_grid(data["data"][seg_id], grid, outliers=outliers, ymin=-2000, ymax=2000)
        save_plot(plot_path, name=tag + '_raw_channels')

    ## preprocess
    if not preload:
        d_out = process_mne(data=data["data"][seg_id], channels=channels, sr=sr_raw, ch_types='ecog',
                            plot_path=plot_path, data_name=tag, bad=outliers, sr_post=sr_post, n_smooth=1)
    else:
        d_out = {}
        band = 'hfb'
        d_out['hfb'] = np.load(str(Path(proc_path)/f'{tag}_car_{band}_{sr_post}Hz.npy'))

    ## events
    t_events_d = np.round(np.array(event_sample_ids)/(sr_raw/sr_post)).astype(int)
    # plt.figure()
    # plt.plot(d_out['hfb'][:, 5])
    # plt.vlines(x=t_events_d, ymin=1, ymax=4, color='black')
    #
    ## save processed + events as a csv
    if not preload:
        for band in d_out.keys():
            np.save(str(Path(proc_path)/f'{tag}_car_{band}_{sr_post}Hz.npy'), d_out[band])

    #
    ## plot
    if not preload:
        plot_signals_on_grid(d_out['hfb'].T, grid, outliers, ymin=1, ymax=4.5)
        save_plot(plot_path, name=tag + '_hfb_channels')

    return d_out, t_events_d, c_events, grid, outliers

