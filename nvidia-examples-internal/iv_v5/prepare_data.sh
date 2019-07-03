#!/bin/bash

source ./default_parameters.inc
model=IntelligentVoice_en-US_8kHz_40000_general_V5_ASRv5

data=${1:-$WORKSPACE/data/}
datasets=$WORKSPACE/datasets/
models=$WORKSPACE/models/

mkdir -p $data
mkdir -p $models/$model
mkdir -p $datasets/$model

# you might not want to do this for interactive shells.
set -e

if [[ "$SKIP_DATA_DOWNLOAD" -ne "1" ]]; then
  echo ----------- Fetching dataset -----------

  pushd $data
  wget http://sqrl/datasets/speech/data/IntelligentVoice_V5-test.tar.gz
  tar -xzf IntelligentVoice_V5-test.tar.gz
  popd >&/dev/null

  pushd $datasets
  wget http://sqrl/datasets/speech/datasets/IntelligentVoice_V5-dataset.tar.gz 
  tar -xzf IntelligentVoice_V5-dataset.tar.gz 
  sed -i 's@workspace@'"${WORKSPACE}"'@' IntelligentVoice_V5/test_enron/wav.scp
  sed -i 's@workspace@'"${WORKSPACE}"'@' IntelligentVoice_V5/test_enron/wav_conv.scp
  popd >&/dev/null
fi

if [[ "$SKIP_MODEL_DOWNLOAD" -ne "1" ]]; then
  echo ----------- Fetching trained model -----------
  pushd $models >&/dev/null
  wget http://sqrl/datasets/speech/models/IntelligentVoice_en-US_8kHz_40000_general_V5_ASRv5-trained.tar.gz 
  tar -xzf IntelligentVoice_en-US_8kHz_40000_general_V5_ASRv5-trained.tar.gz
  cd IntelligentVoice_en-US_8kHz_40000_general_V5_ASRv5/conf
  sed -i 's@workspace@'"${WORKSPACE}"'@' online.conf
  sed -i 's@workspace@'"${WORKSPACE}"'@' online_cmvn.conf
  sed -i 's@workspace@'"${WORKSPACE}"'@' splice.conf
  sed -i 's@workspace@'"${WORKSPACE}"'@' mfcc.conf
  sed -i 's@workspace@'"${WORKSPACE}"'@' ivector_extractor.conf
  popd >&/dev/null
fi

ln -s ../run_cpu_benchmark.sh
ln -s ../run_benchmark.sh
ln -s ../run_multigpu_benchmark.sh

