'''
What this does:
    - normalize native electrode locations to the MNI space using SPM volume method and IDL script from Mathijs

What is changed:

How to run:
    python run_normec.py \
        -c subject.yml
        
Yml should contain: 
    name: name
    electrode_coordinates_path: filepath to native space electrodes
    anatomy_path: path to native anatomy
    norm_electrode_coordinates_path: filepath where to save MNI coordinates to
    
    

Main_config should contain a list of subjects (subject) to run and general path to individual subject's config files (data_path)
Alternatively, individual config is fed per subject, then config should contain subject id (name)
'''

import sys
sys.path.insert(0, '.')
import os
import yaml
import argparse
from pathlib import Path
from subprocess import call
from tempfile import mkdtemp

def load_config(config_path):
    assert Path.exists(config_path), f'{config_path} does not exist'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(main_config_path):
    main_config =  load_config(Path(main_config_path))
    if 'subject' in main_config.keys():
        for id in main_config['subject']:
            subject_config_file = Path(main_config['data_path']) / f'{id}.yml'
            subject_config = load_config(subject_config_file)
            run_one_subject(subject_config)
    elif 'name' in main_config.keys():
        run_one_subject(main_config)
    else:
        raise NotImplementedError


def run_one_subject(config, run_bash=True, save_log=True):
    idl_tool = ''
    subject = config['name']
    ecoord_path = config['electrode_coordinates_path']
    anat_path = config['anatomy_path']

    script_sh_path = Path(__file__).resolve().parent / 'shell_mni_idl'
    script_sh_path.mkdir(parents=True, exist_ok=True)

    assert Path.exists(Path(ecoord_path)), f'{ecoord_path} does not exist'
    assert Path.exists(Path(anat_path)), f'{anat_path} does not exist'

    out_path = Path(config['norm_electrode_coordinates_path'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ecoord_ext = os.path.splitext(ecoord_path)[-1]

    tmp_dir = mkdtemp() # procedure generate temp files, dump them here
    temp_path = os.path.join(tmp_dir, os.path.split(anat_path)[-1])

    if ecoord_ext == '.mat':
        idl_str = 'idl -e "normalize_coors_script_mat,' + "'" + ecoord_path + "'," + \
                  "'elecmatrix'," + "'" + temp_path.replace('.gz', '') + "'," + \
                  "'" + str(out_path) + "'" + '"'
        idl_tool = 'normalize_coors_script_mat'
    elif ecoord_ext == '.txt':
        idl_str = 'idl -e "normalize_coors_script,' + "'" + ecoord_path + "'," + \
                  "'" + temp_path.replace('.gz', '') + "'," + "'" + str(out_path) + "'" + '"'
        idl_tool = 'normalize_coors_script'
    else:
        raise NotImplementedError

    #script_sh_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'shell_mni_idl')
    with open(script_sh_path / f'{subject}_mni.sh', 'w') as f:
        f.write('cp ' + anat_path + ' ' + temp_path + '\n')
        if os.path.splitext(temp_path)[-1] == '.gz': f.write('gzip -d ' + temp_path +'\n')
        f.write(idl_str + '\n')
        if os.path.splitext(temp_path)[-1] == '.gz': f.write('gzip ' + temp_path.replace('.gz', '') + '\n')

    print('Bash script written: ' + subject + '_mni.sh')
    f.close()

    if run_bash:
        with open(script_sh_path / f'{subject}_mni.sh', 'rb') as file:
            script = file.read()
        rc = call(script, shell=True)
        if save_log:
            with open(out_path.parent / 'readme', 'w') as f:
                f.write('Electrode coordinate file used: ' + ecoord_path + '\n')
                f.write('Anatomy file used: ' + anat_path + '\n')
                f.write('Tool used: IDL ' + idl_tool + ' by Mathijs Raemaekers \n')
            print('Log written: to ' + str(out_path.parent) + ' readme')
            f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subject config file')
    parser.add_argument('--config_path', '-c', type=str, help='Input config file', default='')
    args = parser.parse_args()

    main(args.config_path)
