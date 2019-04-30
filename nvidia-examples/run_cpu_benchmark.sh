#!/bin/bash
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

#set local model parameters
source ./default_parameters.inc
#set global model parameters
source ../default_parameters.inc

export KALDI_ROOT=${KALDI_ROOT:-/opt/kaldi}

result_path=/tmp/ls-results

#CPU_THREADS=1
echo "CPU Threads: $CPU_THREADS"
echo "iterations: $ITERATIONS"
echo "file-limit: $FILE_LIMIT"
echo "beam=$BEAM"
echo "lattice-beam=$LATTICE_BEAM"
echo "max-active=$MAX_ACTIVE"
echo "MODEL_PATH=$MODEL_PATH"
echo "DATASET_PATH=$DATASET_PATH"
echo "DATASETS=$DATASETS"

DECODERS="online2-wav-nnet3-latgen-faster"
DECODER_PATH="online2bin"

spk2utt="spk2utt"
wavscp="wav_conv.scp"
WAVSCP=$wavscp

model_path_hash=$(echo $MODEL_PATH | md5sum | cut -c1-8)

mkdir -p $result_path

# copy vocabulary locally as lowercase (see below caveat for comment on this)
cat $MODEL_PATH/words.txt | tr '[:upper:]' '[:lower:]' > $result_path/words.txt

declare -A rtfx

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

    if [ $FILE_LIMIT > 0 ]; then
      cat $DATASET_PATH/$test_set/$WAVSCP | head -n $FILE_LIMIT &> $DATASET_PATH/$test_set/${WAVSCP}.trnc
      wavscp=${WAVSCP}.trnc
    fi

    # run the target decoder with the current dataset
    echo "Running $decoder decoder on $test_set in $CPU_THREADS processes..."

    for (( i=0; i<$CPU_THREADS; i++ )); do
      stdbuf -o 0 numactl --physcpubind=$i $KALDI_ROOT/src/$DECODER_PATH/$decoder --frame-subsampling-factor=3 \
        --config="$MODEL_PATH/conf/online.conf" --frames-per-chunk=264 --online=false  \
        --max-mem=100000000 --beam=$BEAM --lattice-beam=$LATTICE_BEAM --acoustic-scale=1.0 --determinize-lattice=true --max-active=$MAX_ACTIVE \
        $MODEL_PATH/final.mdl \
        $MODEL_PATH/HCLG.fst \
        "ark:$DATASET_PATH/$test_set/$spk2utt" \
        "scp:$DATASET_PATH/$test_set/$wavscp" \
        "ark:|gzip -c > $result_path/lat.$i.$decoder.$test_set.gz" 2>&1 &> $log_file.$i&
    done

    echo "Waiting on processes to finish"
    wait

    # output processing speed from debug log
    rtft=0
   
    for (( i=0; i<$CPU_THREADS; i++ )); do
      echo "  CPU: $i"
      cp $log_file.$i $log_file

      rtf=$(cat $log_file | grep "real-time" | cut -d " " -f 11)
      rtf=`echo "scale=2;1.0 / $rtf" | bc`
      rtft=`echo "scale=2;$rtft + $rtf" | bc`

      # convert lattice to transcript
      $KALDI_ROOT/src/latbin/lattice-best-path \
        "ark:gunzip -c $result_path/lat.$i.$decoder.$test_set.gz |"\
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
      echo "  RTFx=$rtf"
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
    echo "Test set: $test_set, $CPU_THREADS threads RTF: $rtft"
    rtfx[$test_set]=$rtft
  done
done

echo "BENCHMARK SUMMARY ($CPU_THREADS threads):"
for decoder in $DECODERS ; do
  for test_set in $DATASETS ; do
    echo "    test_set: $test_set, RTFx: ${rtfx[$test_set]}"
  done
done
exit $fail
