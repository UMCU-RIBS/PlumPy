'''
Load/write info from/to files
'''
import json
import warnings
import numpy as np
import yaml
import pandas as pd
import sys
sys.path.insert(0, '/home/julia/Documents/Python/Blackrock/Python-Utilities/')
import brpylib
from pathlib import Path

def load_config(config_path):
    if type(config_path) == str:
        config_path = Path(config_path)
    assert Path.exists(config_path), f'{config_path} does not exist'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
    
def write_config(save_path, config):
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def data2csv(save_path, save_name, **kwargs):
    save_path.mkdir(parents=True, exist_ok=True)
    out2 = pd.DataFrame()
    for k, v in kwargs.items():
        out2[k] = v
    out2.to_csv(save_path / f'{save_name}.csv')


def dict2json(save_path, save_name, d):
    save_path.mkdir(parents=True, exist_ok=True)
    if type(d) == dict:
        d_ = d
    elif hasattr(d, '__dict__'):
        d_ = d.__dict__
    else:
        raise NotImplementedError
    with open(save_path / f'{save_name}.json', 'w') as fp:
        json.dump(d_, fp, indent=4)

def load_grid(grid_path):
    #grid_path = subject['grid_map']
    channels = pd.read_csv(grid_path, header=None)
    channels = [int(i.strip('ch')) for i in channels[0]]
    grid = np.array(channels).reshape(-1, 8)
    return grid

def load_processed(task, run, config):
    params = config['preprocess']
    d_out = {}
    for band in params['bands']:
        tmp_name = f'{task}_{run}_{params["reference"]}_{band}_{params["target_sampling_rate"]}Hz.npy'
        d_out[band] = np.load(str(Path(config['data_path'])/ tmp_name))
    events = pd.read_csv(str(Path(config['data_path']) / f'{task}_{run}_events.csv'))
    return d_out, events

def load_blackrock(nev_path, elec_ids='all', start_time_s=0, data_time_s='all'):
    '''

    :param nev_path: str: path to the nev file (nsx files should be there too)
    :param elec_ids: list or 'all': list of elec_ids to extract (e.g., [13]) or "all".
    :param start_time_s: float: starting time for data extraction in seconds
    :param data_time_s: float or 'all': duration of data to return (e.g., 30.0) or "all".
    :return:
        cont_data: dict with raw data in cont_data['data']
        all_data: dict, trigger information
        units: list of str per channel: uV, used in plots
    '''

    datafile_nev = Path(nev_path)
    assert datafile_nev.exists(), f'{datafile_nev} does not exist'
    for x in [2, 3]:
        if Path(nev_path.replace('nev', f'ns{x}')).exists():
            nsx_path = nev_path.replace('nev', f'ns{x}')
            datafile_nsx = Path(nsx_path)
            warnings.warn(f'Loading ns{x}. Double-check if this is correct.')
            break

    #assert datafile_nsx.exists(), f'{datafile_nsx} does not exist'

    nsx_file = brpylib.NsxFile(str(datafile_nsx))
    cont_data = nsx_file.getdata(elec_ids, start_time_s, data_time_s, downsample=1, full_timestamps=True)
    nsx_file.close()

    nev_file = brpylib.NevFile(str(datafile_nev))
    all_data = nev_file.getdata(elec_ids)
    nev_file.close()
    units = [i['Units'] for i in nsx_file.extended_headers] # microvolts

    return cont_data, all_data, units

