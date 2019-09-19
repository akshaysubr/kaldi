#!/bin/bash

WORKSPACE=${WORKSPACE:-"/workspace/"}

MODEL_NAME=${MODEL_NAME:-"ja-JP"}
MODEL_PATH=${MODEL_PATH:-"${WORKSPACE}/models/iv/${MODEL_NAME}/"}

DATASET=${DATASET:-"${WORKSPACE}/datasets/iv/${MODEL_NAME}/"}

GPU=${GPU:-0}
GPU_THREADS=${GPU_THREADS:-3}
COPY_THREADS=${COPY_THREADS:-3}

COMPUTE_CER=${COMPUTE_CER:-true}
BEAM=${BEAM:-15}
LATTICE_BEAM=${LATTICE_BEAM:-2.5}
MAX_ACTIVE=${MAX_ACTIVE:-6000}
MAIN_Q_CAPACITY=${MAIN_Q_CAPACITY:-30000}
AUX_Q_CAPACITY=${AUX_Q_CAPACITY:-400000}

#query GPU memory
gpu_memory=`nvidia-smi -q -i $GPU | grep -A1 "FB Memory" | grep Total | tr -s " " | cut -d " " -f 4`
if [ $gpu_memory -ge 32000 ]; then
  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-400}
  BATCH_DRAIN_SIZE=${BATCH_DRAIN_SIZE:-40}
elif [ $gpu_memory -ge 16000 ]; then
  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-200}
  BATCH_DRAIN_SIZE=${BATCH_DRAIN_SIZE:-20}
elif [ $gpu_memory -ge 8000 ]; then
  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-100}
  BATCH_DRAIN_SIZE=${BATCH_DRAIN_SIZE:-10}
elif [ $gpu_memory -ge 4000 ]; then
  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-50}
  BATCH_DRAIN_SIZE=${BATCH_DRAIN_SIZE:-5}
else
  echo "ERROR not enough GPU memory to run benchmark."
  exit 1;
fi


