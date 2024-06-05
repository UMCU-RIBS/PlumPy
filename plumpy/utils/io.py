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

def load_grid(id, data_path):
    from riverfern.utils.io import load_by_id
    subject = load_by_id(id, data_path)
    grid_path = subject['grid_map']
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

def load_datapaths(path):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    return pd.read_csv(path)

def select_datafiles(hdr, data_files):
    '''Run the script with the datapaths.'''
    from plumpy.utils.general import to_list
    header = hdr.copy()
    for k in header.keys():
        header[k] = to_list(header[k])
    # Select your task
    if 'task' not in header.keys() or not header['task']:
        print('--------- You did not subselect a task. All files will now be selected')
        datafiles = data_files
    else:
        datafiles = data_files[data_files['task'].isin(header['task'])]
        if datafiles.empty:
            datafiles = data_files
            print('--------- Your task is not in this dataset. You continue with all files.')
        else:
            print(f'--------- Selecting task: {header["task"]}')

    # Select your brainFunction
    if 'brainFunction' not in header.keys() or not header['brainFunction']:
        print('--------- You did not subselect a brainFunction. All files will now be selected')
        data_files = datafiles
    else:
        data_files = datafiles[datafiles['brainFunction'].isin(header['brainFunction'])]
        if datafiles.empty:
            data_files = datafiles
            print('--------- Your brainFunction is not in this dataset. You continue with all files.')
        else:
            print(f'--------- Selecting brainFunction: {header["brainFunction"]}')

    # Select your recording App
    if 'app' not in header.keys() or not header['app']:
        print('--------- You did not subselect a recording app. All files will now be selected')
        datafiles = data_files
    else:
        datafiles = data_files[data_files['app'].isin(header['app'])]
        if datafiles.empty:
            datafiles = data_files
            print('--------- Your app is not in this dataset. You continue with all files.')
        else:
            print(f'--------- Selecting recording app: {header["app"]}')

    # Select your session(s)
    if 'session' not in header.keys() or not header['session']:
        print('--------- You did not subselect a session. All sessions will now be loaded')
        data_files = datafiles
    else:
        if not header['session']:
            print('--------- Selecting all sessions')
            data_files = datafiles
        else:
            data_files = datafiles[datafiles['session'].isin(header['session'])]
            if datafiles.empty:
                data_files = datafiles
                print('--------- Your session is not in this dataset. You continue with all files.')
            else:
                print(f'--------- Selecting session(s): {header["session"]}')

    # Make subselection of feedback tasks or not
    if 'feedback' not in header.keys() or not header['feedback']:
        print(
            '--------- You did not subselect if you want Feedback/ no Feedback tasks. All tasks will now be loaded')
        datafiles = data_files
    else:
        datafiles = data_files[data_files['feedback'].isin(header['feedback'])]
        if data_files.empty:
            datafiles = data_files
            if header['feedback'] == 1:
                print('--------- You do not have FB tasks in this dataset. You continue with all files.')
            elif header['feedback'] == 0:
                print('--------- You do not have no FB tasks in this dataset. You continue with all files.')
        else:
            if header['feedback'] == 1:
                print('--------- Selecting tasks with feedback on')
            elif header['feedback'] == 0:
                print('--------- Selecting tasks with feedback off')

    return datafiles

def get_data(config):
    from riverfern.utils.io import load_by_id
    subject = load_by_id(config['subject'], config['data_path'])
    data_files = load_datapaths(subject['data_paths'])
    datafiles = select_datafiles(config['header'], data_files)
    return datafiles


