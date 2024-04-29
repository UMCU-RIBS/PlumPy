'''
How to run:
    python classify_14nav.py \
        -c /Fridge/users/julia/project_corticom/cc2/14nav/cc2_14nav.yml
'''

import sys
sys.path.insert(0, '.')
sys.path.insert(0, '/home/julia/Documents/Python/RiverFErn')
import argparse
import numpy as np
import pickle
import pandas as pd
from scipy import stats
from pathlib import Path
from riverfern.dataset.Dataset import Dataset, Events
from riverfern.ml.Scaler import Scaler
from riverfern.dataset.Epochs import Epochs
from riverfern.ml.OptimizedSVM import OptimizedSVM
from riverfern.ml.ParallelCV import ParallelCV_SVM
from riverfern.utils.plots import plot_svm_scores, plot_optuna_opt_history, plot_optuna_opt_param
from riverfern.utils.io import load_config, load_by_id
import matplotlib
matplotlib.use('Agg')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def classify(config_path):
    config = load_config(Path(config_path))
    params = config['classify']

    for id in config['subject']:
        subject = load_by_id(id, data_path=str(Path(config['data_path']).parent))
        print(subject['name'])
        gen_save_path = Path(config['save_path']) / f'train_{len(config["classify"]["runs"])}runs' / params['strategy'] / '_'.join(config['features'])
        gen_plot_path = Path(config['plot_path']) / f'train_{len(config["classify"]["runs"])}runs'/ params['strategy'] / '_'.join(config['features'])

        for sr in params['sampling_rate']:
            for tmin, tmax in zip(params['epochs']['tmin'], params['epochs']['tmax']):
                save_path = gen_save_path / f'{str(sr)}Hz_{str(tmin)}_{str(tmax)}'
                plot_path = gen_plot_path / f'{str(sr)}Hz_{str(tmin)}_{str(tmax)}'
                save_path.mkdir(parents=True, exist_ok=True)
                plot_path.mkdir(parents=True, exist_ok=True)

                raw = {k: [] for k in config['classify']['runs']}
                events = {k: [] for k in config['classify']['runs']}
                scaler = {k: [] for k in config['classify']['runs']}
                epochs = {k: [] for k in config['classify']['runs']}
                for run in config['classify']['runs']:
                    variant = config['order'][run]
                    raw[run] = Dataset(id=subject['name'],
                                  input_paths={k: v for k, v in config['feature_paths'][sr][run].items() if k in config['features']},
                                  channel_paths={k: config['channel_paths'][run] for k in config['features']},
                                  sampling_rate=sr)
                    events[run] = Events(id=subject['name'],
                                    events_path=config['events_paths'][run],
                                    select_path=config['selected_events'][variant],
                                    units='seconds')

                    scaler[run] = Scaler(mean_paths={k: v for k, v in config['mean_paths'][sr][run].items() if k in config['features']},
                                    scale_paths={k: v for k, v in config['scale_paths'][sr][run].items() if k in config['features']})
                    raw_transformed = scaler[run].transform(raw[run])

                    epochs[run] = Epochs(raw_transformed, events[run],
                                    tmin=tmin,
                                    tmax=tmax)
                    epochs[run].data2array()

                # concatenate runs
                events_all = Events(id=subject)
                events_all.dataframe = pd.concat([events[i].dataframe for i in config['classify']['runs']]).reset_index(drop=True)
                events_all.update_label_encoder()

                epochs_data = np.vstack(epochs[i].data for i in config['classify']['runs'])  # will be different per run if maxlen!
                events_data = np.hstack(events[i].data for i in config['classify']['runs'])

                # create a classifier instance
                svm = OptimizedSVM(id=subject['name'],
                                   cBounds=(params['optimized_svm']['c_min'], params['optimized_svm']['c_max']),
                                   n_optuna_trials=params['optimized_svm']['n_optuna_trials'],
                                   kernel=params['optimized_svm']['kernel'],
                                   forceCPU=params['optimized_svm']['force_cpu'])

                # run cv
                cv = ParallelCV_SVM(cv_type='group')
                cv.make_groups(events_data)
                #best = np.array([67, 28, 104, 103, 102, 101, 113, 111, 110, 109, 99, 97, 105]) - 1
                #best = np.array([61, 65, 66, 67, 69, 72, 87, 96, 97, 101, 102, 103, 104, 105, 109, 110, 111, 113]) - 1
                best = np.array([67, 68, 69, 70, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113]) - 1
                scores, predictions, targets, clfs = cv.run(svm, epochs_data[..., best], events_data)
                scores2, predictions, targets, clfs = cv.run(svm, np.mean(epochs_data[..., best], 1, keepdims=True), events_data)

                print('_'.join(events_all.classes))
                print('Sampling rate: ' + str(sr) + ', tmin: ' + str(tmin) + ', tmax: ' + str(tmax))
                print('Median CV accuracy: ' + str(np.median(scores)) + '+-' + str(stats.median_abs_deviation(scores)))
                print('Mean CV accuracy: ' + str(np.mean(scores)) + '+-' + str(np.std(scores)))
                print('Median CV accuracy avg time: ' + str(np.median(scores2)) + '+-' + str(stats.median_abs_deviation(scores2)))
                print('Mean CV accuracy avg time: ' + str(np.mean(scores2)) + '+-' + str(np.std(scores2)))
                print('done')

                # # plot and save
                # plot_svm_scores(scores, plot_path)
                # out = pd.DataFrame()
                # out['accuracy'] = scores
                # out['n_timepoints'] = epochs_data.data.shape[1]
                # out['fixed_duration'] = epochs[run].fixed_duration
                # out['sampling_rate'] = sr
                # out['tmin_tmax'] = f'{str(tmin)}_{str(tmax)}'
                # out.to_csv(save_path / 'train_runs_results.csv')
                #
                # svm.train(epochs_data, events_data, groups=cv.groups, n_jobs=1)
                # plot_optuna_opt_history(svm.optuna_study, plot_path)
                # plot_optuna_opt_param(svm.optuna_study, plot_path)
                # pickle.dump(svm.clf, open(save_path / 'svm.p', 'wb'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for decoding')
    parser.add_argument('--config_path', '-c', type=str, help='Input config file', default='')
    args = parser.parse_args()

    classify(args.config_path)


