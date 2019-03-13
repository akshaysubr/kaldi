#!/bin/bash

declare -A DATA_PATH;
declare -A MODELS_PATH;
declare -A DATASETS;

DATA_PATH["librispeech"]="/workspace/datasets/LibriSpeech/"
DATA_PATH["aspire"]="/workspace/datasets/aspire/"

MODELS_PATH["librispeech"]="/workspace/models/LibriSpeech/"
MODELS_PATH["aspire"]="/workspace/models/aspire/"

DATASETS["aspire"]="test_clean test_other"
DATASETS["librispeech"]="test_clean test_other"

#key is the parent directory name
key=`basename $PWD`

#set variables for other scripts
model_path=${MODELS_PATH[$key]}
dataset_path=${DATA_PATH[$key]}
datasets=${DATASETS[$key]}

