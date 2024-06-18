'''
How to run:
    python plumpy/scripts/classify/classify_optimized_svm.py \
        -c /Fridge/bci/data/23-171_CortiCom/F_DataAnalysis/plumpy_configs/config_classify_gestures_optimized_svm.yml
'''

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '/home/julia/Documents/Python/RiverFErn')
sys.path.insert(0, '/home/julia/Documents/Python/PlumPy')
import argparse
import numpy as np
import pandas as pd
from scipy import stats 
from riverfern.dataset.Dataset import Dataset, Events
from riverfern.ml.Scaler import Scaler
from riverfern.dataset.Epochs import Epochs
from riverfern.ml.OptimizedSVM import OptimizedSVM
from riverfern.ml.ParallelCV import ParallelCV_SVM
from plumpy.scripts.quality_checks import run_dqc
from plumpy.utils.io import load_config, get_data
import matplotlib
matplotlib.use('Agg')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def classify(all_data, all_events, params):
    tmin = params['epochs']['tmin']
    tmax = params['epochs']['tmax']
    runs = range(len(all_data))

    raw = {k: [] for k in runs}
    events = {k: [] for k in runs}
    scaler = {k: [] for k in runs}
    epochs = {k: [] for k in runs}

    for run in runs:
        # put data into the dataset: dictionary with keys = frequency bands
        raw[run] = Dataset(id=run,
                           inputs=all_data[run],
                           sampling_rate=params['sampling_rate'],
                           downsample_factor=params['downsample_factor'])
        # put data into events: dataframe with events in events[run].dataframe
        events[run] = Events(id=run,
                             events_df=all_events[run],
                             selection=params['classes'],
                             units='seconds')

        # normalize input: zscore
        scaler[run] = Scaler(data=raw[run].data)
        raw_transformed = scaler[run].transform(raw[run])

        # select trials = epochs based on events
        epochs[run] = Epochs(raw_transformed, events[run],
                             tmin=tmin,
                             tmax=tmax,
                             column_onset='times_sec')
        # dictionary of data to 4d array: trials x timepoints x bands x electrodes
        epochs[run].data2array()

    # concatenate runs
    events_all = Events(id=None)
    events_all.dataframe = pd.concat([events[i].dataframe for i in runs]).reset_index(drop=True)
    events_all.update_label_encoder()

    epochs_data = np.vstack([epochs[i].data for i in runs])
    events_data = np.hstack([events[i].data for i in runs])

    # create a classifier instance
    svm = OptimizedSVM(id=params['strategy'],
                       cBounds=(params['optimized_svm']['c_min'], params['optimized_svm']['c_max']),
                       n_optuna_trials=params['optimized_svm']['n_optuna_trials'],
                       kernel=params['optimized_svm']['kernel'],
                       forceCPU=params['optimized_svm']['force_cpu'])

    # run cv
    cv = ParallelCV_SVM(cv_type='group')
    cv.make_groups(events_data)

    scores, predictions, targets, clfs = cv.run(svm, epochs_data, events_data)
    scores2, predictions, targets, clfs = cv.run(svm, np.mean(epochs_data, 1, keepdims=True), events_data)

    # report results
    print('_'.join(events_all.classes))
    print(f'Sampling rate: {params['sampling_rate']}, tmin: {tmin}, tmax: {tmax}')
    print(f'Median CV accuracy: {np.median(scores)} +- {stats.median_abs_deviation(scores)}')
    print(f'Mean CV accuracy: {np.mean(scores)} +- {np.std(scores)}')
    print(f'Median CV accuracy avg time: {np.round(np.median(scores2), 2)}+-{np.round(stats.median_abs_deviation(scores2), 2)}')
    print(f'Mean CV accuracy avg time: {np.round(np.mean(scores2), 2)}+- {np.round(np.std(scores2), 2)}')
    print('Done')

def main(config_file):
    # get data
    config = load_config(config_file)
    data_files = get_data(config)
    data, events = [], []

    # process data one by one
    for i, rec in data_files.iterrows():
        print(rec)
        output = run_dqc(rec,
                         config['preprocess'],
                         preload=True,
                         save_dir=config['data_path'],
                         plot_dir=None)
        data.append(output[0])
        events.append(output[1])

    # classify
    config['classify']['sampling_rate'] = config['preprocess']['target_sampling_rate']
    classify(data, events, config['classify'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for decoding')
    parser.add_argument('--config_path', '-c', type=str, help='Input config file', default='')
    args = parser.parse_args()

    main(args.config_path)
