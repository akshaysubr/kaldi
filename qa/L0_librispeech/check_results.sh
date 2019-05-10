#!/bin/bash

source gold.inc

NUM_GPUS=`nvidia-smi -L | wc -l`
NUM_GPUS=1

DATASETS="test_clean test_other"

FAIL=0

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



echo "CHECKING RESULTS"
for dataset in $DATASETS;
do
  testname="${NUM_GPUS}x${GPU_NAME}_${dataset}"
  echo "testname=$testname"
  for (( d = 0 ; d < $NUM_GPUS ; d++ )); do
    WER=`cat /tmp/ls-results.$d/log.batched-wav-nnet3-cuda.$dataset.out | grep "%WER" | cut -d " " -f 2`
    PERF=`cat /tmp/ls-results.$d/log.batched-wav-nnet3-cuda.$dataset.out | grep Overall | grep Aggregate | cut -d ":" -f 8 | cut -d " " -f 2`

    EWER=${EXPECTED_WER[$dataset]}
    EPERF=${EXPECTED_PERF[$testname]}

    echo "    dataset=$dataset, GPU=$d: " 
    echo "         WER=$WER, Expected=$EWER"
    PASS=`echo "$WER <= $EWER" | bc`
    if [ $PASS -ne "1" ]; then
      echo "              Error:  WER rate ($WER) greater than  $EWER"
      FAIL=1
    fi
    echo "         PERF=$PERF, Expected=$EPERF"
    PASS=`echo "$PERF >= $EPERF" | bc`
    if [ $PASS -ne "1" ]; then
      echo "              Error:  PERF ($PERF) less than than  $EPERF"
      FAIL=1
    fi
  done
done

if [ $FAIL -eq "0" ]; then
  echo "All WER and PERF tests passed"
fi
exit $FAIL

