'''
'''
import pandas as pd
import sys
sys.path.insert(0, './src/')
sys.path.insert(0, '/home/julia/Documents/Python/PlumPy')
from timeit import default_timer as timer
from plumpy.utils.io import load_config, load_blackrock, load_grid
from plumpy.utils.plots import *
from plumpy.sigproc.general import calculate_rms, ind2sec, sec2ind
from plumpy.sigproc.process import process_mne
pd.set_option('display.max_rows', 500)


##

def run_dqc(recording, config, preload=False, plot=False):
    ## set params
    task = config['header']['task']
    tag = str(Path(recording['filename']).name)
    name = config['subject']
    sr_post = config['preprocess']['target_sampling_rate']
    grid = load_grid(name, config['subj_path'])
    plot_path = str(Path(config['plot_path']) / name / task)
    proc_path = str(Path(config['data_path']) / name / task)

    ## load data
    start = timer()
    data, events, units = load_blackrock(recording['filename'])
    elapsed_time = timer() - start
    print(f'It took {elapsed_time} seconds to load the data')
    sr_raw = data['samp_per_s']
    if plot:
        plot_data(data, events, units=units)
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

    ## plot rms
    rms = calculate_rms(data["data"][seg_id])
    if plot:
        plot_on_grid(grid, rms, label='RMS', colormap='viridis_r')
        save_plot(plot_path, name=tag + '_raw_rms')

    ## plot means
    means = np.mean(data["data"][seg_id], 1)
    if plot:
        plot_on_grid(grid, means, label='Mean', colormap='viridis_r')
        save_plot(plot_path, name=tag + '_raw_mean')

    ## plot channels on a grid
    stds = np.std(data["data"][seg_id], 1)
    outliers = np.where(stds > np.percentile(stds, 95))[0]
    #outliers = np.where(stds > 200)[0]
    if plot:
        if not preload: # slow plots, only plot once
            plot_signals_on_grid(data["data"][seg_id], grid, outliers=outliers, ymin=-2000, ymax=2000)
            save_plot(plot_path, name=tag + '_raw_channels')

    ## events
    t_events_d = np.round(np.array(event_sample_ids)/(sr_raw/sr_post)).astype(int)
    t_events_sec = np.array([ind2sec(i, sr_post) for i in t_events_d])
    #t_events_sec = np.array([ind2sec(i, sr_raw) for i in event_sample_ids]) # more accurate but worse accuracy?
    #t_events_d = np.array([sec2ind(i, sr_post) for i in t_events_sec])
    events_df = pd.DataFrame({'events':c_events, f'samples_{sr_post}Hz': t_events_d, 'times_sec' : t_events_sec})
    # plt.figure()
    # plt.plot(d_out['hfb'][:, 5])
    # plt.vlines(x=t_events_d, ymin=1, ymax=4, color='black')

    ## preprocess
    #channels = pd.read_csv(subj_cfg['grid_map'], header=None)
    #channels = [int(i.strip('ch')) for i in channels[0]]
    channels = list(np.sort(grid.flatten()))

    if task == 'words':
        if recording['session'] < 9 and recording['brainFunction'] == 'NavWords_8-14':
            task_events = events_df['events'].values - 7
            task_events[task_events == 24] = 31
            task_events[0] = 200
            task_events[-1] = 201
            task_events[1] = 1
            task_events[2] = 50
            events_df['events'] = task_events
    events_df.to_csv(str(Path(proc_path) / f'{tag}_events.csv'))


    if not preload:
        # process
        d_out = process_mne(data=data["data"][seg_id], channels=channels, sr=sr_raw, ch_types='ecog',
                            plot_path=plot_path, data_name=tag, bad=outliers, sr_post=sr_post, n_smooth=1,
                            freqs=config['preprocess']['bands'])
        # save
        for band in d_out.keys():
            np.save(str(Path(proc_path)/f'{tag}_car_{band}_{sr_post}Hz.npy'), d_out[band])
        if task == 'words':
            if recording['session'] < 9 and recording['brainFunction'] == 'NavWords_8-14':
                # different codes were used before the dynamic cue
                task_events = events_df['events'].values - 7
                task_events[task_events == 24] = 31
                task_events[0] = 200
                task_events[-1] = 201
                task_events[1] = 1
                task_events[2] = 50
                events_df['events'] = task_events
        events_df.to_csv(str(Path(proc_path) / f'{tag}_events.csv'))

    else:
        d_out = {}
        band = 'hfb'
        d_out['hfb'] = np.load(str(Path(proc_path)/f'{tag}_car_{band}_{sr_post}Hz.npy'))
        events_df = pd.read_csv(str(Path(proc_path)/f'{tag}_events.csv'))

    ## plot
    if plot:
        if not preload: # slow plots, only plot once
            plot_signals_on_grid(d_out['hfb'].T, grid, outliers, ymin=1, ymax=4.5)
            save_plot(plot_path, name=tag + '_hfb_channels')

    return d_out, events_df, outliers

