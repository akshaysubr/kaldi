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
cd /workspace/nvidia-examples/swbd-eng

bash -ex ./prepare_data.sh

NUM_GPUS=`nvidia-smi -L | wc -l`


NUM_PROCESSES=$NUM_GPUS EXPECTED_WER=${GOLD_WER["swbd-eng"]} EXPECTED_PERF=${GOLD_PERF["${GPU_NAME}x${NUM_GPUS}_swbd-eng"]} DATASET=/workspace/datasets/swbd-eng/ bash -e ./run_benchmark.sh


popd
