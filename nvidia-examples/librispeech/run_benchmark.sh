#!/bin/bash

echo "Usage: $0 <model_path> <dataset_path> <result_path> [gpu_threads] [cpu_threads] [max_batch_size] [batch_drain_size] [iterations] [file_limit] [beam] [lattice_beam] [max_active]"

DECODERS="batched-wav-nnet3-cuda"
DATASETS="test_clean test_other"

export KALDI_ROOT=${KALDI_ROOT:-/opt/kaldi}

model_path=${1:-/workspace/models/LibriSpeech/}
dataset_path=${2:-/workspace/datasets/LibriSpeech/}
result_path=${3:-/tmp/ls-results}
gpu_threads=${4:-2}
cpu_threads=${5:-80}
max_batch_size=${6:-100}
batch_drain_size=${7:-20}
iterations=${8:-10}
file_limit=${9:--1}
beam=${10:-10}
lattice_beam=${11:-7}
max_active=${12:-10000}

generate_lattices=0

#NVPROF="nvprof -f -o profile.out"

let worker_threads="$cpu_threads - $gpu_threads"
if [ $worker_threads -lt 0 ]; then
  worker_threads=0
fi

echo "GPU Threads: $gpu_threads CPU Threads: $cpu_threads Worker Threads: $worker_threads Batch Size: $max_batch_size Batch Drain: $batch_drain_size Iterations: $iterations FileLimit: $file_limit beam=$beam lattice-beam=$lattice_beam max-active=$max_active"

wavscp="wav_conv.scp"

model_path_hash=$(echo $model_path | md5sum | cut -c1-8)
mkdir -p $result_path

# copy vocabulary locally as lowercase (see below caveat for comment on this)
cat $model_path/words.txt | tr '[:upper:]' '[:lower:]' > $result_path/words.txt

for test_set in $DATASETS ; do
  mkdir $result_path/$test_set
  echo "Generating new reference transcripts for model and dataset..."
  cat $dataset_path/$test_set/text | tr '[:upper:]' '[:lower:]' > $result_path/$test_set/text
  oovtok=$(cat $result_path/words.txt | grep "<unk>" | awk '{print $2}')

  $KALDI_ROOT/egs/wsj/s5/utils/sym2int.pl --map-oov $oovtok -f 2- $result_path/words.txt $result_path/$test_set/text > $result_path/$test_set/text_ints_$model_path_hash 2> /dev/null
done

for decoder in $DECODERS ; do
  for test_set in $DATASETS ; do
    log_file="$result_path/log.$decoder.$test_set.out"

    path="cudadecoderbin"
    cuda_flags="--cuda-use-tensor-cores=true --iterations=$iterations --max-tokens-per-frame=500000 --cuda-memory-proportion=.5 --max-batch-size=$max_batch_size --cuda-control-threads=$gpu_threads --batch-drain-size=$batch_drain_size --cuda-worker-threads=$worker_threads"
    if [ "$generate_lattices" -eq "1" ]; then
      cuda_flags="$cuda_flags --generate-lattices=true"
    fi

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
    else
      echo "  Decoding completed successfully."
    fi
  done
done
