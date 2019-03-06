#!/bin/bash

cd /workspace/nvidia-examples/librispeech

SKIP_DOWNLOAD=1 SKIP_FLAC2WAV=1 ./prepare_data.sh /data/speech
exec ./run_benchmark.sh
