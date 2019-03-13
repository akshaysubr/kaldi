#!/bin/bash
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

#sets datasets
source ../examples.inc

if [ $# -ge 1 ]; then
  num_gpus=$1
else
  num_gpus=`nvidia-smi -L | wc -l`
fi

total_gpus=`nvidia-smi -q | grep "Product Name" | wc -l`
total_threads=`cat /proc/cpuinfo | awk '/^processor/{print $3}' | wc -l`
threads_per_gpu=`echo $total_threads/$num_gpus | bc`

for dataset in $datasets; do
  echo "Running $dataset on $num_gpus GPUs with $threads_per_gpu threads per GPU"
  
  for (( d = 0 ; d < $num_gpus ; d++ )); do
    ./run_benchmark.sh $d $dataset 2 $threads_per_gpu &> output.$d&
  done
  
  wait

  TOTAL_RTF=0
  for (( d = 0 ; d < $num_gpus ; d++ )); do
    RTF=`cat output.$d | grep Aggregate | tail -n 1 | tr -s " " | cut -d " " -f 11`
    TOTAL_RTF=`echo "$RTF + ${TOTAL_RTF}" | bc`
    echo "GPU: $d RTF: $RTF"
  done
  AVERAGE_RTF=`echo "scale=4; ${TOTAL_RTF} / $num_gpus" | bc -l`
  echo "Total RTF: ${TOTAL_RTF} Average RTF: $AVERAGE_RTF"
done
