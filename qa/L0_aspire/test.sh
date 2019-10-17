#!/bin/bash

source gold.inc

GPU_NAME=UNKNOWN
if [[ $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep "Tesla V100" | wc -l) -gt 0 ]]; then
	GPU_NAME="V100"
elif [[ $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep "Tesla P100" | wc -l) -gt 0 ]]; then
	GPU_NAME="P100"
elif [[ $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep "Tesla P40" | wc -l) -gt 0 ]]; then
	GPU_NAME="P40"
elif [[ $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep "Tesla T4" | wc -l) -gt 0 ]]; then
	GPU_NAME="T4"
fi

pushd .
cd /workspace/nvidia-examples/aspire

bash -ex ./prepare_data.sh
#SKIP_DATA_DOWNLOAD=1 SKIP_FLAC2WAV=1 SKIP_MODEL_DOWNLOAD=1 bash -ex ./prepare_data.sh

NUM_GPUS=`nvidia-smi -L | wc -l`


ITERATIONS=5 NUM_PROCESSES=$NUM_GPUS EXPECTED_WER=${GOLD_WER["test_clean"]} EXPECTED_PERF=${GOLD_PERF["${GPU_NAME}x${NUM_GPUS}_test_clean"]} DATASET=/workspace/datasets/aspire/test_clean/ bash -e ./run_benchmark.sh

ITERATIOS=5 NUM_PROCESSES=$NUM_GPUS EXPECTED_WER=${GOLD_WER["test_other"]} EXPECTED_PERF=${GOLD_PERF["${GPU_NAME}x${NUM_GPUS}_test_other"]} DATASET=/workspace/datasets/aspire/test_other/ bash -e ./run_benchmark.sh

popd
