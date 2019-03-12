#!/bin/bash

source gold.inc

num_gpus=`nvidia-smi -L | wc -l`
datasets="test_clean test_other"

fail=0

echo "CHECKING RESULTS"
for dataset in $datasets;
do
  for (( d = 0 ; d < $num_gpus ; d++ )); do
    WER=`cat /tmp/ls-results.$d/log.batched-wav-nnet3-cuda.$dataset.out | grep "%WER" | cut -d " " -f 2`
    pass=`echo "${WER} > ${EXPECTED_WER[$dataset]}" | bc`

    echo "    dataset=$dataset, GPU=$d, WER=$WER, Expected=${EXPECTED_WER[$dataset]}"
    if [ $pass -ne "0" ]; then
      echo "      Error:  WER rate $WER greater than  ${EXPECTED_WER[$dataset]}"
      fail=1
    fi
  done
done

exit $fail

