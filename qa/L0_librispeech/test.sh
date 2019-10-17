#!/bin/bash

source gold.inc

export MAX_BATCH_SIZE=150
#lowering these to avoid potential OOM errors
export BATCH_DRAIN_SIZE=10

if [[ $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep "Tesla V100" | wc -l) -gt 0 ]]; then
  export MAX_BATCH_SIZE=250
fi

pushd .
cd /workspace/nvidia-examples/librispeech

./prepare_data.sh
#SKIP_DATA_DOWNLOAD=1 SKIP_FLAC2WAV=1 SKIP_MODEL_DOWNLOAD=1 ./prepare_data.sh

NUM_GPUS=`nvidia-smi -L | wc -l`


NUM_PROCESSES=$NUM_GPUS EXPECTED_WER=${GOLD_WER["test_clean"]} EXPECTED_PERF=${GOLD_PERF["${GPU_NAME}x${NUM_GPUS}_test_clean"]} DATASET=/workspace/datasets/LibriSpeech/test_clean/ bash -e ./run_benchmark.sh

NUM_PROCESSES=$NUM_GPUS EXPECTED_WER=${GOLD_WER["test_other"]} EXPECTED_PERF=${GOLD_PERF["${GPU_NAME}x${NUM_GPUS}_test_other"]} DATASET=/workspace/datasets/LibriSpeech/test_other/ bash -e ./run_benchmark.sh

popd
