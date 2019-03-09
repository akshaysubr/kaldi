#!/bin/bash

cd /workspace/nvidia-examples/librispeech

SKIP_DOWNLOAD=1 SKIP_FLAC2WAV=1 ./prepare_data.sh /data/speech
./run_benchmark.sh
./run_multigpu_benchmark.sh

#TODO validate results
