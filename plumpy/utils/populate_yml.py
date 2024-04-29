from plumpy.utils.io import load_config, write_config
from plumpy.utils.general import to_list
from pathlib import Path
import argparse


def populate_yml(config_path):
    config = load_config(config_path)
    task = config['task']
    params = config['preprocess']
    srs = to_list(params['downsample_sampling_rate'])
    refs = to_list(params['reference'])
    bands = list(params['bands'].keys())
    data_path = Path(config['data_path'])
    data_path.mkdir(parents=True, exist_ok=True)

    for i in ['feature_paths', 'mean_paths', 'scale_paths']:
        config[i] = {}
        for sr in srs:
            config[i][sr] = {}
            for run in config['include_runs']:
                config[i][sr][run] = {}
                for band in bands:
                    for ref in refs:
                        if i == 'feature_paths':
                            config[i][sr][run][band] = str(
                                data_path / f'{task}_{run}_{ref}_{band}_{sr}Hz_nobad.npy')
                        elif i == 'mean_paths':
                            config[i][sr][run][band] = str(
                                data_path / f'{task}_{run}_{ref}_{band}_{sr}Hz_nobad_precomputed_mean.npy')
                        elif i == 'scale_paths':
                            config[i][sr][run][band] = str(
                                data_path / f'{task}_{run}_{ref}_{band}_{sr}Hz_nobad_precomputed_scale.npy')
                        assert Path(
                            config[i][sr][run][band]).exists(), f'Path does not exist:{config[i][sr][run][band]}'
    #
    for i in ['events_paths']:
        config[i] = {}
        for run in config['include_runs']:
            config[i][run] = {}
            if i == 'events_paths':
                config[i][run] = str(data_path / f'{task}_{run}_all_words_onsets.csv')
                assert Path(config[i][run]).exists(), f'Path does not exist:{config[i][run]}'

    for i in ['channel_paths']:
        config[i] = {}
        for run in config['include_runs']:
            config[i][run] = str(data_path / f'{task}_{run}_channel_indices.csv')
    #
    write_config(config_path, config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for decoding')
    parser.add_argument('--config_path', '-c', type=str, help='Input config file', default='')
    args = parser.parse_args()

    populate_yml(args.config_path)