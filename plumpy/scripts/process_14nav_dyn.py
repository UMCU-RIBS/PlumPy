'''
python plumpy/scripts/process_14nav_dyn.py -c /Fridge/users/julia/project_corticom/cc2/config_14nav.yml
'''
import sys
import argparse
import pandas as pd
sys.path.insert(0, './src/')
sys.path.insert(0, '/home/julia/Documents/Python/PlumPy')
from plumpy.scripts.quality_checks import run_dqc
from plumpy.utils.io import load_config
from plumpy.scripts.map_active_vs_rest import map_active_one, map_active_mean
from plumpy.scripts.prepare4classify import prepare4classify
from plumpy.scripts.classify_14nav import classify
pd.set_option('display.max_rows', 500)


##
def main(config_file):
    config = load_config(config_file)
    task = config['task']

    for run in config['include_runs']:
        print(run)
        #data, events, _ = run_dqc(config, task, run, preload=True, plot=False)
        #map_active_one(data, events, config, run, plot=True)

    #map_active_mean(config)
    prepare4classify(config_file)
    classify(config_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input configuration file')
    parser.add_argument('--config_path', '-c', type=str, help='Input config file', default='')
    args = parser.parse_args()

    main(args.config_path)