#!/bin/bash
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

echo "Usage: $0 <GPU_IDX> <data-sets> [gpu_threads] [cpu_threads] [max_batch_size] [batch_drain_size] [iterations] [file_limit] [beam] [lattice_beam] [max_active]"

#sets model_path, dataset_path, datasets
source ../examples.inc

#by default use device 0 unless told otherwise
export gpu=${1:-0}

result_path=/tmp/ls-results.$gpu

DECODERS="batched-wav-nnet3-cuda"

export KALDI_ROOT=${KALDI_ROOT:-/opt/kaldi}
export CUDA_VISIBLE_DEVICES=$gpu

if [ $# -ge 1 ]; then
 datasets=$2
fi

gpu_threads=${3:-2}

#by default use all cores
cpu_threads=${4:-`cat /proc/cpuinfo | awk '/^processor/{print $3}' | wc -l`}

#query GPU memory
gpu_memory=`nvidia-smi -q -i 0 | grep -A1 "FB Memory" | grep Total | tr -s " " | cut -d " " -f 4`

if [ $gpu_memory -ge 16000 ]; then
  max_batch_size=${5:-100}
  batch_drain_size=${6:-20}
elif [ $gpu_memory -ge 8000 ]; then
  max_batch_size=${5:-50}
  batch_drain_size=${6:-10}
elif [ $gpu_memory -ge 4000 ]; then
  max_batch_size=${5:-25}
  batch_drain_size=${6:-5}
else
  echo "ERROR not enough GPU memory to run benchmark."
  exit 1;
fi


iterations=${7:-10}
file_limit=${8:--1}
beam=${9:-10}
lattice_beam=${10:-7}
max_active=${11:-10000}

#NVPROF="nvprof -f -o profile.out"

let worker_threads="$cpu_threads - $gpu_threads"

#must always have at least one worker thread
if [ $worker_threads -lt 0 ]; then
    worker_threads=1
fi

echo "GPU: $CUDA_VISIBLE_DEVICES GPU Threads: $gpu_threads CPU Threads: $cpu_threads Worker Threads: $worker_threads Batch Size: $max_batch_size Batch Drain: $batch_drain_size Iterations: $iterations FileLimit: $file_limit beam=$beam lattice-beam=$lattice_beam max-active=$max_active"

wavscp="wav_conv.scp"

model_path_hash=$(echo $model_path | md5sum | cut -c1-8)
mkdir -p $result_path

# copy vocabulary locally as lowercase (see below caveat for comment on this)
cat $model_path/words.txt | tr '[:upper:]' '[:lower:]' > $result_path/words.txt

for test_set in $datasets ; do
  mkdir $result_path/$test_set
  echo "Generating new reference transcripts for model and dataset..."
  cat $dataset_path/$test_set/text | tr '[:upper:]' '[:lower:]' > $result_path/$test_set/text
  oovtok=$(cat $result_path/words.txt | grep "<unk>" | awk '{print $2}')

  $KALDI_ROOT/egs/wsj/s5/utils/sym2int.pl --map-oov $oovtok -f 2- $result_path/words.txt $result_path/$test_set/text > $result_path/$test_set/text_ints_$model_path_hash 2> /dev/null
done

fail=0
for decoder in $DECODERS ; do
  for test_set in $datasets ; do
    log_file="$result_path/log.$decoder.$test_set.out"

    path="cudadecoderbin"
    cuda_flags="--cuda-use-tensor-cores=true --iterations=$iterations --max-tokens-per-frame=500000 --cuda-memory-proportion=.5 --max-batch-size=$max_batch_size --cuda-control-threads=$gpu_threads --batch-drain-size=$batch_drain_size --cuda-worker-threads=$worker_threads"

    # run the target decoder with the current dataset
    echo "Running $decoder decoder on $test_set$trunc [$threads threads]..."
    stdbuf -o 0 $NVPROF $KALDI_ROOT/src/$path/$decoder $cuda_flags --frame-subsampling-factor=3 \
      --config="$model_path/conf/online.conf" --frames-per-chunk=264  --file-limit=$file_limit\
      --max-mem=100000000 --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=1.0 --determinize-lattice=true --max-active=$max_active \
      $model_path/final.mdl \
      $model_path/HCLG.fst \
      "scp:$dataset_path/$test_set/$wavscp" \
      "ark:|gzip -c > $result_path/lat.$decoder.$test_set.gz" 2>&1 | tee $log_file

    if [ $? -ne 0 ]; then
      echo "  ERROR encountered while decoding. Check $log_file"
      fail=1
      continue
    fi

    # output processing speed from debug log
    rtf=$(cat $log_file | grep RealTimeX | cut -d' ' -f 3-)
    echo "  $rtf"

    # convert lattice to transcript
    $KALDI_ROOT/src/latbin/lattice-best-path \
      "ark:gunzip -c $result_path/lat.$decoder.$test_set.gz |"\
      "ark,t:|gzip -c > $result_path/trans.$decoder.$test_set.gz" >>$log_file 2>&1

    # calculate wer
    $KALDI_ROOT/src/bin/compute-wer --mode=present \
      "ark:$result_path/$test_set/text_ints_$model_path_hash" \
      "ark:gunzip -c $result_path/trans.$decoder.$test_set.gz |" >>$log_file 2>&1

    # output accuracy metrics
    wer=$(cat $log_file | grep "%WER")
    ser=$(cat $log_file | grep "%SER")
    scored=$(cat $log_file | grep "Scored")
    echo "  $wer"
    echo "  $ser"
    echo "  $scored"

    # ensure all expected utterances were processed
    expected_sentences=$(cat $dataset_path/$test_set/$wavscp | wc -l)
    actual_sentences=$(echo $scored | awk '{print $2}')
    echo "  Expected: $expected_sentences, Actual: $actual_sentences"
    if [ $expected_sentences -ne $actual_sentences ]; then
      echo "  Error: did not return expected number of utterances. Check $log_file"
      fail=1
    else
      echo "  Decoding completed successfully."
    fi
  done
done


echo "BENCHMARK SUMMARY:"
for decoder in $DECODERS ; do
  for test_set in $datasets ; do
    log_file="$result_path/log.$decoder.$test_set.out"
    echo "    test_set: $test_set"
    echo "        `cat $log_file | grep 'Overall:  Aggregate Total Time' | cut -d ")" -f 3 |  awk '{$1=$1};1'`"
    echo "        `cat $log_file | grep 'WER'`"
    echo "        `cat $log_file | grep 'SER'`"
    echo "        `cat $log_file | grep 'Scored'`"
  done
done

exit $fail
