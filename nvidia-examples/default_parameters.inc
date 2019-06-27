#!/bin/bash

#set global default parameters only if the variable is not already set
GPU=${GPU:-0}
GPU_THREADS=${GPU_THREADS:-2}
CPU_THREADS=${CPU_THREADS:-`cat /proc/cpuinfo | awk '/^processor/{print $3}' | wc -l`}
WORKER_THREADS=${WORKER_THREADS:-$(($CPU_THREADS - $GPU_THREADS))}
ITERATIONS=${ITERATIONS:-10}
FILE_LIMIT=${FILE_LIMIT:--1}
BEAM=${BEAM:-10}
LATTICE_BEAM=${LATTICE_BEAM:-7}
MAX_ACTIVE=${MAX_ACTIVE:-10000}
MAIN_Q_CAPACITY=${MAIN_Q_CAPACITY:-40000}
AUX_Q_CAPACITY=${AUX_Q_CAPACITY:-500000}
GPU_FEATURE=${GPU_FEATURE:-true}

#query GPU memory
gpu_memory=`nvidia-smi -q -i $GPU | grep -A1 "FB Memory" | grep Total | tr -s " " | cut -d " " -f 4`
if [ $gpu_memory -ge 16000 ]; then
  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-180}
  BATCH_DRAIN_SIZE=${BATCH_DRAIN_SIZE:-15}
elif [ $gpu_memory -ge 8000 ]; then
  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-90}
  BATCH_DRAIN_SIZE=${BATCH_DRAIN_SIZE:-8}
elif [ $gpu_memory -ge 4000 ]; then
  MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-40}
  BATCH_DRAIN_SIZE=${BATCH_DRAIN_SIZE:-4}
else
  echo "ERROR not enough GPU memory to run benchmark."
  exit 1;
fi

#must always have at least one worker thread
if [ ${WORKER_THREADS} -lt 1 ]; then
    WORKER_THREADS=1
fi
