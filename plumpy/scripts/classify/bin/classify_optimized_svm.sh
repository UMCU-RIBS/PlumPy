#!/bin/bash
# bash classify_optimized_svm.sh \
# -c /Fridge/bci/data/23-171_CortiCom/F_DataAnalysis/plumpy_configs/classify/config_classify_gestures_svm_subset_channels.yml
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -c|--config)
      config="$2"
      shift # past argument
      shift # past value
      ;;

    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

echo 'Welcome to PlumPy'
echo 'Activating environment... '
source /home/julia/miniconda3/bin/activate
conda activate mne
echo 'Done '

echo 'Running classification... '
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
python $PARENT_DIR/classify_optimized_svm.py \
    -c $config
    
echo 'Done '

echo 'Deactivating environment... '
conda deactivate
echo 'Done '
echo 'Shutting down now'

