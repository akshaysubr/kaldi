#!/bin/bash

pushd .
cd /workspace/nvidia-examples/librispeech

SKIP_DOWNLOAD=1 SKIP_FLAC2WAV=1 ./prepare_data.sh /data/speech/data
bash -e ./run_benchmark.sh
bash -e ./run_multigpu_benchmark.sh

popd
bash -e ./check_results.sh
