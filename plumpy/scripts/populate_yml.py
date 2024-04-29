from riverfern.utils.io import load_config, load_by_id, write_config
from pathlib import Path
import argparse


def main(config_path):
    ##
    srs = [100, 50, 10]
    refs = ['car']
    bands = {'hfb': '60-90'}

    config = load_config(Path(config_path))
    globals = load_by_id('cc2', data_path=str(Path(config['data_path']).parent))
    print(globals['name'])
    task = config['task']

    data_path = Path(globals['data_path'])
    data_path.mkdir(parents=True, exist_ok=True)

    for i in ['feature_paths', 'mean_paths', 'scale_paths']:
        config[i] = {}
        for sr in srs:
            config[i][sr] = {}
            for run in globals['include_runs']:
                config[i][sr][run] = {}
                for band in bands:
                    for ref in refs:
                        if i == 'feature_paths':
                            config[i][sr][run][band] = str(data_path / f'{task}_{run}_{ref}_{band}_{sr}Hz_nobad.npy')
                        elif i == 'mean_paths':
                            config[i][sr][run][band] = str(data_path / f'{task}_{run}_{ref}_{band}_{sr}Hz_nobad_precomputed_mean.npy')
                        elif i == 'scale_paths':
                            config[i][sr][run][band] = str(data_path / f'{task}_{run}_{ref}_{band}_{sr}Hz_nobad_precomputed_scale.npy')
                        assert Path(config[i][sr][run][band]).exists(), f'Path does not exist:{config[i][sr][run][band]}'
#
    for i in ['events_paths']:
        config[i] = {}
        for run in globals['include_runs']:
            config[i][run] = {}
            if i == 'events_paths':
                config[i][run] = str(data_path / f'{task}_{run}_all_prep_onsets.csv')
                assert Path(config[i][run]).exists(), f'Path does not exist:{config[i][run]}'

    for i in ['channel_paths']:
        config[i] = {}
        for run in globals['include_runs']:
            config[i][run] = str(data_path / f'{task}_{run}_channel_indices.csv')
#
    write_config(config_path, config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for decoding')
    parser.add_argument('--config_path', '-c', type=str, help='Input config file', default='')
    args = parser.parse_args()

    main(args.config_path)