'''
How to run:
    python /home/julia/Documents/Python/PlumPy/plumpy/scripts/classify_14nav.py \
        -c /Fridge/users/julia/project_corticom/data/config_classify.yml
'''

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '/home/julia/Documents/Python/RiverFErn')
sys.path.insert(0, '/home/julia/Documents/Python/PlumPy')
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from riverfern.dataset.Dataset import Dataset, Events
from riverfern.ml.Scaler import Scaler
from riverfern.dataset.Epochs import Epochs
from riverfern.ml.OptimizedSVM import OptimizedSVM
from riverfern.ml.ParallelCV import ParallelCV_SVM
from riverfern.utils.plots import plot_svm_scores, plot_optuna_opt_history, plot_optuna_opt_param
from plumpy.scripts.quality_checks import run_dqc
from plumpy.scripts.prepare4classify import prepare4classify
from plumpy.utils.io import load_config, get_data
from plumpy.utils.general import to_list
import matplotlib
matplotlib.use('Agg')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def classify(config_path):
    config = load_config(Path(config_path))
    params = config['classify']
    tmin = params['epochs']['tmin']
    tmax = params['epochs']['tmax']
    sr = params['sampling_rate']
    features = to_list(params['bands'])
    variant = params['variant']
    runs = [i for i in config['dyn_runs'] if config['order'][i] == variant]
    # runs = [i for i in config['dyn_runs'] if config['order'][i] == variant]

    name = config["subject"]
    print(name)
    gen_save_path = Path(config['save_path']) / f'train_{len(runs)}runs' / params['strategy'] / '_'.join(features)
    gen_plot_path = Path(config['plot_path']) / f'train_{len(runs)}runs'/ params['strategy'] / '_'.join(features)
    save_path = gen_save_path / f'{str(sr)}Hz_{str(tmin)}_{str(tmax)}'
    plot_path = gen_plot_path / f'{str(sr)}Hz_{str(tmin)}_{str(tmax)}'
    save_path.mkdir(parents=True, exist_ok=True)
    plot_path.mkdir(parents=True, exist_ok=True)
    task = config['task']
    task_info = load_config(f'{config["meta_path"]}/task_{task}.yml')

    raw = {k: [] for k in runs}
    events = {k: [] for k in runs}
    scaler = {k: [] for k in runs}
    epochs = {k: [] for k in runs}


    for word in list(task_info['codes'][variant].values())[:-1]:
        selection = [word, 'rust']
        for run in runs:
            raw[run] = Dataset(id=name,
                          input_paths={k: v for k, v in config['feature_paths'][sr][run].items() if k in features},
                          channel_paths={k: config['channel_paths'][run] for k in features},
                          sampling_rate=sr)
            events[run] = Events(id=name,
                            events_path=config['events_paths'][run],
                            #selection=params['classes'][variant],
                            selection=selection,
                            units='seconds')

            scaler[run] = Scaler(mean_paths={k: v for k, v in config['mean_paths'][sr][run].items() if k in features},
                            scale_paths={k: v for k, v in config['scale_paths'][sr][run].items() if k in features})
            raw_transformed = scaler[run].transform(raw[run])

            epochs[run] = Epochs(raw_transformed, events[run], tmin=tmin, tmax=tmax)
            epochs[run].data2array()

        # concatenate runs
        events_all = Events(id=name)
        events_all.dataframe = pd.concat([events[i].dataframe for i in runs]).reset_index(drop=True)
        events_all.update_label_encoder()

        epochs_data = np.vstack(epochs[i].data for i in runs)  # will be different per run if maxlen!
        events_data = np.hstack(events[i].data for i in runs)

        # create a classifier instance
        svm = OptimizedSVM(id=name,
                           cBounds=(params['optimized_svm']['c_min'], params['optimized_svm']['c_max']),
                           n_optuna_trials=params['optimized_svm']['n_optuna_trials'],
                           kernel=params['optimized_svm']['kernel'],
                           forceCPU=params['optimized_svm']['force_cpu'])

        # run cv
        cv = ParallelCV_SVM(cv_type='group')
        cv.make_groups(events_data)

        scores, predictions, targets, clfs = cv.run(svm, epochs_data, events_data)
        scores2, predictions, targets, clfs = cv.run(svm, np.mean(epochs_data, 1, keepdims=True), events_data)

        print('_'.join(events_all.classes))
        print(f'Subject: {name}, sampling rate: {sr}, tmin: {tmin}, tmax: {tmax}')
        print(f'Median CV accuracy: {np.median(scores)} +- {stats.median_abs_deviation(scores)}')
        print(f'Mean CV accuracy: {np.mean(scores)} +- {np.std(scores)}')
        print(f'Median CV accuracy avg time: {np.round(np.median(scores2), 2)}+-{np.round(stats.median_abs_deviation(scores2), 2)}')
        print(f'Mean CV accuracy avg time: {np.round(np.mean(scores2), 2)}+- {np.round(np.std(scores2), 2)}')
        print('done')

def main(config_file):
    # get data
    config = load_config(config_file)
    data_files = get_data(config)

    # process data one by one
    for i, rec in data_files.iterrows():
        print(rec)
        data, events, _ = run_dqc(rec, config, preload=False, plot=False)

    # classify
    prepare4classify(config_file)
    classify(config_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for decoding')
    parser.add_argument('--config_path', '-c', type=str, help='Input config file', default='')
    args = parser.parse_args()

    main(args.config_path)
