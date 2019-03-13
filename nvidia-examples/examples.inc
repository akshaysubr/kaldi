#!/bin/bash

declare -A DATA_PATH;
declare -A MODELS_PATH;
declare -A DATASETS;

DATA_PATH=(
  ["librispeech"]="/workspace/datasets/LibriSpeech/"
)

MODELS_PATH=(
  ["librispeech"]="/workspace/models/LibriSpeech/"
)

DATASETS=(
  ["librispeech"]="test_clean test_other"
)

#key is the parent directory name
key=`basename $PWD`

#set variables for other scripts
model_path=${MODELS_PATH[$key]}
dataset_path=${DATA_PATH[$key]}
datasets=${DATASETS[$key]}

