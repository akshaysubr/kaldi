#!/bin/bash

source gold.inc

num_gpus=`nvidia-smi -L | wc -l`
data_sets="test_clean test_other"

fail=0

for dataset in $datasets;
do
  for (( d = 0 ; d < $num_gpus ; d++ )); do
    WER=`cat /tmp/ls-results.$d | grep "%WER" | cut -d " " -f 2`
  	pass=`echo "${WER} > ${EXPECTED_WER[$test_set]}" | bc`
  	
		echo "dataset=$dataset, WER=$WER, Expected=${EXPECTED_WER[$dataset]}"
    if [ $pass -ne "0" ]; then
      echo "  Error:  WER rate $WER great than  ${EXPECTED_WER[$test_set]}"
      fail=1
    else
      echo "WER rate passed"
    fi
  done
done

exit $fail
