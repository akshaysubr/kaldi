#!/bin/bash

cd /workspace/nvidia-examples/librispeech

SKIP_DOWNLOAD=1 SKIP_FLAC2WAV=1 ./prepare_data.sh /data/speech
bash -ex ./run_benchmark.sh
bash -ex ./run_multigpu_benchmark.sh
bash -ex ./check_results.sh
