#!/bin/bash

datasets="test_clean test_other"
if [ $# -ge 1 ]; then
  NUM_GPUS=$1
else
  NUM_GPUS=`nvidia-smi -L | wc -l`
fi

total_threads=`cat /proc/cpuinfo | awk '/^processor/{print $3}' | wc -l`

threads_per_gpu=`echo $total_threads/$NUM_GPUS | bc`

for dataset in $datasets; do
  echo "Running $dataset on $NUM_GPUS GPUs with $threads_per_gpu threads per GPU"
  for (( gpu = 0 ; gpu < $NUM_GPUS ; gpu++ )); do
    ./run_benchmark.sh $gpu $dataset 2 $threads_per_gpu &> output.$gpu&
  done

  wait

  TOTAL_RTF=0
  for (( gpu = 0 ; gpu < $NUM_GPUS ; gpu++ )); do

    RTF=`cat output.$gpu | grep Aggregate | tail -n 1 | cut -d " " -f 11`
    TOTAL_RTF=`echo $RTF + $TOTAL_RTF | bc`
    echo "GPU: $gpu RTF: $RTF"
  done
  AVERAGE_RTF=`echo "scale=4; $TOTAL_RTF / $NUM_GPUS" | bc -l`
  echo "Total RTF: $TOTAL_RTF Average RTF: $AVERAGE_RTF"
done
