#!/bin/bash

datasets="test_clean"
#datasets="test_clean test_other"
if [ $# -ge 1 ]; then
  num_gpus=$1
else
  num_gpus=`nvidia-smi -L | wc -l`
fi

total_gpus=`nvidia-smi -q | grep "Product Name" | wc -l`

total_threads=`cat /proc/cpuinfo | awk '/^processor/{print $3}' | wc -l`

threads_per_gpu=`echo $total_threads/$num_gpus | bc`
half_gpus=`echo "($total_gpus+1)/2" | bc`

for dataset in $datasets; do
  echo "Running $dataset on $num_gpus GPUs with $threads_per_gpu threads per GPU"
  for (( d = 0 ; d < $num_gpus ; d++ )); do
    #swizzle GPUs to distributed across PCI-E lanes
    #this assumes GPUs are distributed evenly across sockets
    mod=`echo $d%$half_gpus*2 | bc`
    half=`echo $d/$half_gpus | bc`
    gpu=`echo $mod+$half | bc`
    #echo "d=$d mod=$mod half=$half gpu=$gpu"
    ./run_benchmark.sh $gpu $dataset 2 $threads_per_gpu &> output.$d&
  done

  wait

  TOTAL_RTF=0
  for (( d = 0 ; d < $num_gpus ; d++ )); do
    RTF=`cat output.$d | grep Aggregate | tail -n 1 | cut -d " " -f 11`
    TOTAL_RTF=`echo $RTF + $TOTAL_RTF | bc`
    echo "GPU: $d RTF: $RTF"
  done
  AVERAGE_RTF=`echo "scale=4; $TOTAL_RTF / $num_gpus" | bc -l`
  echo "Total RTF: $TOTAL_RTF Average RTF: $AVERAGE_RTF"
done
