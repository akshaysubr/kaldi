#!/bin/bash

ln -s /data/speech/models /workspace/models
ln -s /data/speech/data /workspace/data

pushd .
cd /workspace/nvidia-examples/aspire

SKIP_DATA_DOWNLOAD=1 SKIP_FLAC2WAV=1 SKIP_MODEL_DOWNLOAD=1 ./prepare_data.sh

#run test with bad parameters to cause lots of overflow
MAX_TOKENS_PER_FRAME=50000 DATA_SETS='test_other' bash -e ./run_benchmark.sh &> output.log
tail -n 100 output.log

MAX_TOKENS_PER_FRAME=450000 bash -e ./run_benchmark.sh
MAX_TOKENS_PER_FRAME=450000 bash -e ./run_multigpu_benchmark.sh 1

popd
bash -e ./check_results.sh
