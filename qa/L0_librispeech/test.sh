#!/bin/bash
ln -s /data/speech/models /workspace/models
ln -s /data/speech/data /workspace/data

pushd .
cd /workspace/nvidia-examples/librispeech

SKIP_DATA_DOWNLOAD=1 SKIP_FLAC2WAV=1 SKIP_MODEL_DOWNLOAD=1 ./prepare_data.sh
bash -e ./run_benchmark.sh
#bash -e ./run_multigpu_benchmark.sh

popd
bash -e ./check_results.sh
