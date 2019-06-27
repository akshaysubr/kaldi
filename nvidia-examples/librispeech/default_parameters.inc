#!/bin/bash

MODEL_PATH=${MODEL_PATH:-"/workspace/models/LibriSpeech/"}
DATASET_PATH=${DATASET_PATH:-"/workspace/datasets/LibriSpeech/"}
DATASETS=${DATASETS:-"test_clean test_other"}

GPU=${GPU:-0}
GPU_THREADS=${GPU_THREADS:-2}
COPY_THREADS=${COPY_THREADS:-0}

BEAM=${BEAM:-10}
LATTICE_BEAM=${LATTICE_BEAM:-7}
MAX_ACTIVE=${MAX_ACTIVE:-10000}
MAIN_Q_CAPACITY=${MAIN_Q_CAPACITY:-30000}
AUX_Q_CAPACITY=${AUX_Q_CAPACITY:-400000}

#query GPU memory
gpu_memory=`nvidia-smi -q -i $GPU | grep -A1 "FB Memory" | grep Total | tr -s " " | cut -d " " -f 4`
if [ $gpu_memory -ge 32000 ]; then
  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-600}
  BATCH_DRAIN_SIZE=${BATCH_DRAIN_SIZE:-80}
elif [ $gpu_memory -ge 16000 ]; then
  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-300}
  BATCH_DRAIN_SIZE=${BATCH_DRAIN_SIZE:-40}
elif [ $gpu_memory -ge 8000 ]; then
  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-150}
  BATCH_DRAIN_SIZE=${BATCH_DRAIN_SIZE:-20}
elif [ $gpu_memory -ge 4000 ]; then
  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-75}
  BATCH_DRAIN_SIZE=${BATCH_DRAIN_SIZE:-10}
else
  echo "ERROR not enough GPU memory to run benchmark."
  exit 1;
fi


