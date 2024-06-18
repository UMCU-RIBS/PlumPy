'''
'''
import typing
import warnings
import pandas as pd
import sys

import plumpy

sys.path.insert(0, './src/')
sys.path.insert(0, '/home/julia/Documents/Python/PlumPy')
from timeit import default_timer as timer
from plumpy.utils.io import load_config, load_blackrock, load_grid
from plumpy.utils.general import to_list
from plumpy.utils.plots import *
from plumpy.sigproc.general import calculate_rms, ind2sec, sec2ind
from plumpy.sigproc.process import process_mne
pd.set_option('display.max_rows', 500)


##
def get_task_event_names(task: str,
                         brainFunction: str,
                         events_df: pd.DataFrame,
                         add_dyn_cue_val: int = None,
                         column_name: str = 'text') -> pd.DataFrame:
    """
    Add names of the individual events to the events dataframe. Pick this information up from the task configuration
    file. By default, assigns the names not to the first entry of the code (typically, preparation cue) but to the
    second one (typically, go cue). If dynamic cue is used, then the assignment is shifted further down for all events
    but "rust".
    Args:
        task: name of the task
        brainFunction: name of the brain function the task addresses
        events_df: dataframe with events information, includes codes and timings
        add_dyn_cue_val: shift assigning names to codes by the specified value. Typically would be 1
        column_name: column name to assign names to in the resulting dataframe

    Returns:
        events_df: resulting dataframe, copy of the events_df with an additional column for event names

    """
    import plumpy
    task_config_name = Path(plumpy.PLUMPY_CONFIG_DIR) / f'task-{task}_brainFunction-{brainFunction}.yml'
    task_config = load_config(task_config_name)
    events_df[column_name]= 0
    for k, v in task_config['codes'].items():
        indices = events_df.loc[events_df['events'] == k].index
        indices = indices + 1
        if add_dyn_cue_val is not None and v != 'rust':
            indices = indices + add_dyn_cue_val
        events_df.loc[indices, 'text'] = v
    return events_df

def run_dqc(recording: pd.Series | typing.Dict,
            params: typing.Dict,
            preload: bool = False,
            save_dir: str = None,
            plot_dir: str = None) -> tuple[typing.Dict, pd.DataFrame]:
    """
    Currently loading, processing and basic quality checks are bundled up here.
    Args:
        recording: information about the recording:
            filename: where the data are stored
            task: what task it belongs to
            brainFunction: what function is studied
            session: number of the session
            app: what app was used
            feedback: whether it is a feedback task or no
        params: parameters for preprocessing
        preload: 0: run preprocessing, 1: load preprocessed data from save_dir
        save_dir: dir for saving preprocessed data
        plot_dir: dir for saving plots, if None, no plots are made

    Returns:
        d_out: preprocessed data per frequency band
        events_df: events with timing

    """
    ## set params
    name = recording['subject']
    task = recording['task']
    brainFunction = recording['brainFunction']
    tag = str(Path(recording['filename']).name)
    sr_post = params['target_sampling_rate']
    grid = load_grid(name, plumpy.PLUMPY_CONFIG_DIR)
    if save_dir:
        save_dir = str(Path(save_dir) / name / task)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    if plot_dir:
        plot_dir = str(Path(plot_dir) / name / task)
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
    
    if not preload:

        # load data
        start = timer()
        data, events, units = load_blackrock(recording['filename'])
        elapsed_time = timer() - start
        print(f'It took {elapsed_time} seconds to load the data')
        sr_raw = data['samp_per_s']
        if plot_dir:
            plot_data(data, events, units=units)
            save_plot(plot_dir, name=tag + '_raw_triggers')

        seg_id = 0
        t_data = data["data_headers"][seg_id]["Timestamp"] / data["samp_per_s"]
        t_events = np.array(events['digital_events']['TimeStamps']) / data["samp_per_s"]
        c_events = np.array(events['digital_events']['UnparsedData'])

        # find sample_ids that correspond to markers
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx], idx

        event_samples, event_sample_ids = zip(*[find_nearest(t_data, i) for i in t_events])
        # verify
        # plt.figure()
        # plt.plot(data["data"][seg_id][5])
        # plt.vlines(x=event_sample_ids, ymin=-400, ymax=400, color='black')

        # rms
        rms = calculate_rms(data["data"][seg_id])
        if plot_dir:
            plot_on_grid(grid, rms, label='RMS', colormap='viridis_r')
            save_plot(plot_dir, name=tag + '_raw_rms')

        # means
        means = np.mean(data["data"][seg_id], 1)
        if plot_dir:
            plot_on_grid(grid, means, label='Mean', colormap='viridis_r')
            save_plot(plot_dir, name=tag + '_raw_mean')

        # plot channels on a grid
        outliers = to_list(params['bad'])
        # stds = np.std(data["data"][seg_id], 1)
        # outliers = np.where(stds > np.percentile(stds, 95))[0]
        # outliers = np.where(stds > 200)[0]
        if plot_dir:
            plot_signals_on_grid(data["data"][seg_id], grid, outliers=outliers, ymin=-2000, ymax=2000)
            save_plot(plot_dir, name=tag + '_raw_channels')

        # preprocess ecog
        channels = list(np.sort(grid.flatten()))
        d_out = process_mne(data=data["data"][seg_id], channels=channels, sr=sr_raw, ch_types='ecog',
                            plot_path=plot_dir, data_name=tag, bad=outliers, reference=params['reference'],
                            sr_post=sr_post, n_smooth=params['smooth_n_samples'], freqs=params['bands'])

        # preprocess events
        t_events_d = np.round(np.array(event_sample_ids) / (sr_raw / sr_post)).astype(int)
        t_events_sec = np.array([ind2sec(i, sr_post) for i in t_events_d])
        # t_events_sec = np.array([ind2sec(i, sr_raw) for i in event_sample_ids]) # more accurate but worse accuracy?
        # t_events_d = np.array([sec2ind(i, sr_post) for i in t_events_sec])
        events_df = pd.DataFrame({'events': c_events, f'samples_{sr_post}Hz': t_events_d, 'times_sec': t_events_sec})

        if task == 'navWords':
            if recording['session'] < 9 and recording['brainFunction'] == '8-14words':
                task_events = events_df['events'].values - 7
                task_events[task_events == 24] = 31
                task_events[0] = 200
                task_events[-1] = 201
                task_events[1] = 1
                task_events[2] = 50
                events_df['events'] = task_events
        get_task_event_names(task, brainFunction, events_df, add_dyn_cue_val=1)  # add 1 for when dynamic cue stops
        print(events_df)

        # save
        if save_dir:
            for band in params['bands']:
                np.save(str(Path(save_dir)/f'{tag}_car_{band}_{sr_post}Hz.npy'), d_out[band])
            events_df.to_csv(str(Path(save_dir) / f'{tag}_events.csv'))

    else:
        try:
            assert Path(save_dir).exists(), f'{save_dir} does not exist'
            d_out = {}
            for band in params['bands']:
                d_out[band] = np.load(str(Path(save_dir) / f'{tag}_car_{band}_{sr_post}Hz.npy'))
            events_df = pd.read_csv(str(Path(save_dir) / f'{tag}_events.csv'))
        except FileNotFoundError:
            warnings.warn(f'Could not load the files. Check {save_dir}')
            raise FileNotFoundError

    ## plot processed
    if plot_dir:
        for band in params['bands']:
            plot_signals_on_grid(d_out[band].T, grid, to_list(params['bad']), ymin=1, ymax=4.5)
            save_plot(plot_dir, name=f'{tag}_{band}_channels')

        plt.figure()
        plt.plot(d_out['hfb'][:, 5])
        plt.vlines(x=events_df[f'samples_{sr_post}Hz'], ymin=1, ymax=4, color='black')
        save_plot(plot_dir, name=f'{tag}_processed_triggers')

    return d_out, events_df

