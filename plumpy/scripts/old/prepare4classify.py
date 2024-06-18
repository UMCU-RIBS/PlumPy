
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, './src/')
sys.path.insert(0, '/home/julia/Documents/Python/PlumPy')
from pathlib import Path
from plumpy.sigproc.general import ind2sec, sec2ind, resample
from plumpy.utils.io import load_config, load_processed
from plumpy.utils.populate_yml import populate_yml
from plumpy.utils.general import to_list
from plumpy.ml.general import zscore


def save4classify(config_path):
    # word onsets
    config = load_config(config_path)
    task = config['task']
    sr_post = config['preprocess']['target_sampling_rate']
    task_info = load_config(f'{config["meta_path"]}/task_{task}.yml')
    data_path = config['data_path']
    params = config['preprocess']
    srs = to_list(params['downsample_sampling_rate'])
    refs = to_list(params['reference'])
    bands = list(params['bands'].keys())
    bad = to_list(config['classify']['bad'])

    for run in config['include_runs']:
        variant = config['order'][run]
        data, events = load_processed(task, run, config)
        task_events = events['events'].values
        t_events_sec = events['times_sec'].values
        all_dict = task_info['codes'][variant]
        onsets = {'text':[], 'xmin':[]}

        for c, i in enumerate(task_events):
            if i in all_dict.keys():
                onsets['text'].append(all_dict[i])
                if all_dict[i] == 'rust':
                    onsets['xmin'].append(t_events_sec[c+1])
                else:
                    if run in config['dyn_runs']:
                        onsets['xmin'].append(t_events_sec[c+2])
                    else:
                        onsets['xmin'].append(t_events_sec[c+1])

        pd.DataFrame(onsets).to_csv(str(Path(data_path)/f'{task}_{run}_all_words_onsets.csv'), index=False)

        # save processed excluding bad channels
        chan_indices = pd.DataFrame({'indices':np.setdiff1d(np.arange(data['hfb'].shape[-1]), np.array(bad))})
        chan_indices.to_csv(str(Path(data_path) / f'{task}_{run}_channel_indices.csv'), index=False)

        d_out = data.copy()
        for ref in refs:
            for band in bands:
                t = np.delete(d_out[band], (bad), axis=1) # TODO: check for len(bad) > 1
                for sr in srs:
                    temp = resample(t, sr, sr_post)
                    np.save(str(Path(data_path)/f'{task}_{run}_{ref}_{band}_{sr}Hz_nobad.npy'), temp)
                    _, scaler2 = zscore(temp, xmin=t_events_sec[task_events == 50][0], duration=5, units='seconds', sr=sr)
                    np.save(str(Path(data_path) / f'{task}_{run}_{ref}_{band}_{sr}Hz_nobad_precomputed_mean.npy'), scaler2.mean_)
                    np.save(str(Path(data_path) / f'{task}_{run}_{ref}_{band}_{sr}Hz_nobad_precomputed_scale.npy'), scaler2.scale_)


def prepare4classify(config_path):
    save4classify(config_path)
    populate_yml(config_path)