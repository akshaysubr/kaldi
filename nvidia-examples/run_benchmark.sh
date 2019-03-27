#!/bin/bash
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

#set local model parameters
source ./default_parameters.inc
#set global model parameters
source ../default_parameters.inc

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU
export KALDI_ROOT=${KALDI_ROOT:-/opt/kaldi}

result_path=/tmp/ls-results.$GPU

echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "CPU Threads: $CPU_THREADS"
echo "cuda-control-threads: $GPU_THREADS"
echo "cuda-worker-threads: $WORKER_THREADS"
echo "batch_size: $MAX_BATCH_SIZE"
echo "batch_drain_size: $BATCH_DRAIN_SIZE"
echo "iterations: $ITERATIONS"
echo "file-limit: $FILE_LIMIT"
echo "leam=$BEAM"
echo "lattice-beam=$LATTICE_BEAM"
echo "max-active=$MAX_ACTIVE"
echo "main-q-capacity=$MAIN_Q_CAPACITY"
echo "aux-q-capacity=$AUX_Q_CAPACITY"
echo "MODEL_PATH=$MODEL_PATH"
echo "DATASET_PATH=$DATASET_PATH"
echo "DATASETS=$DATASETS"

DECODERS="batched-wav-nnet3-cuda"
DECODER_PATH="cudadecoderbin"
#NVPROF="nvprof -f -o profile.out"

wavscp="wav_conv.scp"

model_path_hash=$(echo $MODEL_PATH | md5sum | cut -c1-8)
mkdir -p $result_path

# copy vocabulary locally as lowercase (see below caveat for comment on this)
cat $MODEL_PATH/words.txt | tr '[:upper:]' '[:lower:]' > $result_path/words.txt

for test_set in $DATASETS ; do
  mkdir -p $result_path/$test_set
  echo "Generating new reference transcripts for model and dataset..."
  cat $DATASET_PATH/$test_set/text | tr '[:upper:]' '[:lower:]' > $result_path/$test_set/text
  oovtok=$(cat $result_path/words.txt | grep "<unk>" | awk '{print $2}')

  $KALDI_ROOT/egs/wsj/s5/utils/sym2int.pl --map-oov $oovtok -f 2- $result_path/words.txt $result_path/$test_set/text > $result_path/$test_set/text_ints_$model_path_hash 2> /dev/null
done

fail=0
for decoder in $DECODERS ; do
  for test_set in $DATASETS ; do
    log_file="$result_path/log.$decoder.$test_set.out"

    cuda_flags="--cuda-use-tensor-cores=true --iterations=$ITERATIONS --main-q-capacity=$MAIN_Q_CAPACITY --aux-q-capacity=$AUX_Q_CAPACITY --cuda-memory-proportion=.5 --max-batch-size=$MAX_BATCH_SIZE --cuda-control-threads=$GPU_THREADS --batch-drain-size=$BATCH_DRAIN_SIZE --cuda-worker-threads=$WORKER_THREADS"

    # run the target decoder with the current dataset
    echo "Running $decoder decoder on $test_set..."
    stdbuf -o 0 $NVPROF $KALDI_ROOT/src/$DECODER_PATH/$decoder $cuda_flags --frame-subsampling-factor=3 \
      --config="$MODEL_PATH/conf/online.conf" --frames-per-chunk=264  --file-limit=$FILE_LIMIT\
      --max-mem=100000000 --beam=$BEAM --lattice-beam=$LATTICE_BEAM --acoustic-scale=1.0 --determinize-lattice=true --max-active=$MAX_ACTIVE \
      $MODEL_PATH/final.mdl \
      $MODEL_PATH/HCLG.fst \
      "scp:$DATASET_PATH/$test_set/$wavscp" \
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
    expected_sentences=$(cat $DATASET_PATH/$test_set/$wavscp | wc -l)
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
  for test_set in $DATASETS ; do
    log_file="$result_path/log.$decoder.$test_set.out"
    echo "    test_set: $test_set"
    echo "        `cat $log_file | grep 'Overall:  Aggregate Total Time' | cut -d ")" -f 3 |  awk '{$1=$1};1'`"
    echo "        `cat $log_file | grep 'WER'`"
    echo "        `cat $log_file | grep 'SER'`"
    echo "        `cat $log_file | grep 'Scored'`"
  done
done

exit $fail
