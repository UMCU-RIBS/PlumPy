
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, './src/')
sys.path.insert(0, '/home/julia/Documents/Python/PlumPy')
from pathlib import Path
from plumpy.sigproc.general import ind2sec, sec2ind, resample
from plumpy.utils.io import load_config, load_grid, load_processed
from plumpy.ml.general import zscore

def save4classify(data, events, config, run):
    # word onsets
    task = config['task']
    variant = config['order'][run]
    sr_post = config['preprocess']['target_sampling_rate']
    task_info = load_config(f'{config["meta_path"]}/task_{task}.yml')
    task_events = events['events'].values
    t_events_sec = events['times_sec'].values
    data_path = config['data_path']
    all_dict = task_info['codes'][variant]
    tag = f'{task}_{run}'
    onsets = {'text':[], 'xmin':[]}

    for c, i in enumerate(task_events):
        if i in all_dict.keys():
            onsets['text'].append(all_dict[i])
            if all_dict[i] == 'rust':
                onsets['xmin'].append(t_events_sec[c+1])
            else:
                onsets['xmin'].append(t_events_sec[c+2])

    pd.DataFrame(onsets).to_csv(str(Path(data_path)/f'{tag}_all_words_onsets.csv'), index=False)

    # save processed excluding bad channels
    bad = [120]
    chan_indices = pd.DataFrame({'indices':np.setdiff1d(np.arange(data['hfb'].shape[-1]), np.array(bad))})
    chan_indices.to_csv(str(Path(data_path) / f'{tag}_channel_indices.csv'), index=False)

    d_out = data.copy()
    for band in d_out.keys():
        t = np.delete(d_out[band], (bad), axis=1) # TODO: check for len(bad) > 1
        for sr in [100, 50, 10]:
            temp = resample(t, sr, sr_post)
            np.save(str(Path(data_path)/f'{tag}_car_{band}_{sr}Hz_nobad.npy'), temp)

            _, scaler2 = zscore(temp, xmin=t_events_sec[task_events == 50][0], duration=5, units='seconds', sr=sr)
            np.save(str(Path(data_path) / f'{tag}_car_{band}_{sr}Hz_nobad_precomputed_mean.npy'), scaler2.mean_)
            np.save(str(Path(data_path) / f'{tag}_car_{band}_{sr}Hz_nobad_precomputed_scale.npy'), scaler2.scale_)