#!/bin/bash
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

#this is an example script of tha speech to text pipeline.  
#The purpose of this script is to show how to process wav
#audio using kaldi and produce text transcriptions.
#not all steps are done in the most efficeint or accurate way.  
#In particular segmentation could likely be improved but
#that is currently outside of the scope of this effort which
#is to demonstrate how to process wav's efficiently in kaldi.


function run_benchmark() {

  LOG_FILE="$LOCAL_RESULT_PATH/output.log"
  RTF_FILE="$LOCAL_RESULT_PATH/rtf"
  WER_FILE="$LOCAL_RESULT_PATH/wer"

  # run the target decoder with the current dataset
  echo "    Running $DECODER on $DATASET.  Output log: $LOG_FILE"
  stdbuf -o 0 $NVPROF $DECODER $CUDAFLAGS $CPUFLAGS $FLAGS \
    --config="$MODEL_PATH/conf/online.conf"\
    $MODEL_PATH/final.mdl \
    $MODEL_PATH/HCLG.fst \
    $SPK2UTT \
    "$WAVIN" \
    "ark:|gzip -c > $LOCAL_RESULT_PATH/lat.gz" &> $LOG_FILE

  if [ $? -ne 0 ]; then
    echo "  ERROR encountered while decoding. Check $LOG_FILE"
    exit 1;
  fi

  # output processing speed from debug log
  if [ $USE_GPU -eq 1 ]; then
    RTF=`cat $LOG_FILE | grep RealTimeX | cut -d' ' -f 3-`
  else
    RTF="RTF: `cat $LOG_FILE | grep "real-time factor" | cut -d " " -f 8`"
  fi
  echo "  $RTF" &> $RTF_FILE

  # convert lattice to transcript
  $KALDI_ROOT/src/latbin/lattice-best-path \
    "ark:gunzip -c $LOCAL_RESULT_PATH/lat.gz |"\
    "ark,t:|gzip -c > $LOCAL_RESULT_PATH/trans_int.gz" >>$LOG_FILE 2>&1


  gunzip -c $LOCAL_RESULT_PATH/trans_int.gz | sort -n > $LOCAL_RESULT_PATH/trans_int

  #naively paste output together
  #TODO how do we paste together more wisely?
  #this doesn't work well.  need a way to paste together correctly
  awk '{
      start=match($1,"-[0-9]+-[0-9]+$")-1;
      end=length($1)+2;
      key=substr($1, 1, start);
      trans=substr($0,end);
      transcriptions[key]=transcriptions[key] trans
    }
    END {
     for (key in transcriptions) {
       print key " " transcriptions[key]
     }
    }' \
    $LOCAL_RESULT_PATH/trans_int | sort -n > $LOCAL_RESULT_PATH/trans_int_combined

  #translate ints to words
  $KALDI_ROOT/egs/wsj/s5/utils/int2sym.pl -f 2- $RESULT_PATH/words.txt $LOCAL_RESULT_PATH/trans_int_combined > $LOCAL_RESULT_PATH/trans

  echo "Transcripts output to $LOCAL_RESULT_PATH/trans" 2>&1 >> $LOG_FILE

  #score if necessary
  if [ -f $RESULT_PATH/gold_text_ints ]; then
    # calculate wer
    $KALDI_ROOT/src/bin/compute-wer --mode=present \
      "ark:$RESULT_PATH/gold_text_ints" \
      "ark:$LOCAL_RESULT_PATH/trans_int_combined" >>$LOG_FILE 2>&1

    # calculate character error rate
    if [ "$COMPUTE_CER" = "true" ]; then
      # split trans and gold_text into characters
      cat $LOCAL_RESULT_PATH/trans | perl -CSDA -ane '
        {
          print $F[0];
          foreach $s (@F[1..$#F]) {
            if (($s =~ /\[.*\]/) || ($s =~ /\<.*\>/) || ($s =~ "!SIL")) {
              print " $s";
            } else {
              @chars = split "", $s;
              foreach $c (@chars) {
                print " $c";
              }
            }
          }
          print "\n";
        }' > $LOCAL_RESULT_PATH/trans.chars
      cat $RESULT_PATH/gold_text | perl -CSDA -ane '
        {
          print $F[0];
          foreach $s (@F[1..$#F]) {
            if (($s =~ /\[.*\]/) || ($s =~ /\<.*\>/) || ($s =~ "!SIL")) {
              print " $s";
            } else {
              @chars = split "", $s;
              foreach $c (@chars) {
                print " $c";
              }
            }
          }
          print "\n";
        }' > $LOCAL_RESULT_PATH/text.chars
      # compare with --text
      $KALDI_ROOT/src/bin/compute-wer --text --mode=present \
          "ark:$LOCAL_RESULT_PATH/text.chars" \
	  "ark:$LOCAL_RESULT_PATH/trans.chars" | sed 's/WER/CER/g' >>$LOG_FILE 2>&1
    fi

    # output accuracy metrics
    wer=$(cat $LOG_FILE | grep "%WER")
    cer=$(cat $LOG_FILE | grep "%CER" | cat)
    ser=$(cat $LOG_FILE | grep "%SER" | head -1)
    scored=$(cat $LOG_FILE | grep "Scored")
    echo "  $wer"  > $WER_FILE
    if [ "$COMPUTE_CER" = "true" ]; then
      echo "  $cer"  >> $WER_FILE
    fi
    echo "  $ser"  >> $WER_FILE
    echo "  $scored" >> $WER_FILE

    # ensure all expected utterances were processed
    expected_sentences=$(cat $WAVSCP | wc -l)
    actual_sentences=$(echo $scored | awk '{print $2}')
    echo "  Expected: $expected_sentences, Actual: $actual_sentences" >> $WER_FILE
    if [ $expected_sentences -ne $actual_sentences ]; then
      echo "  Error: did not return expected number of utterances. Check $LOG_FILE" >> $WER_FILE
    else
      echo "  Decoding completed successfully." >> $WER_FILE
    fi
  else
    echo "No gold transcripts found.  Skipping scoring." >> $WER_FILE
  fi
}

#NVPROF="nvprof -f -o profile.out"

#set local model parameters
source ./default_parameters.inc
#set global model parameters
source ../default_parameters.inc

NUM_PROCESSES=${NUM_PROCESSES:-1}
NUM_GPUS=`nvidia-smi -L | wc -l`

THREADS_PER_PROCESS=`echo $CPU_THREADS/$NUM_PROCESSES | bc`

echo "USE_GPU: $USE_GPU"
echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "KALDI_ROOT: $KALDI_ROOT"
echo "WORKSPACE=$WORKSPACE"
echo "DATASET=$DATASET"
echo "MODEL_PATH=$MODEL_PATH"
echo "MODEL_NAME=$MODEL_NAME"

if [ $USE_GPU -eq 1 ]; then
  DECODER=$GPU_DECODER
  CPU_THREADS=$THREADS_PER_PROCESS
  #these are GPU specific parameters
  echo "GPU_FEATURE: $GPU_FEATURE"
  echo "CPU_THREADS: $CPU_THREADS"
  echo "GPU_THREADS: $GPU_THREADS"
  echo "WORKER_THREADS: $WORKER_THREADS"
  echo "MAX_BATCH_SIZE: $MAX_BATCH_SIZE"
  echo "BATCH_DRAIN_SIZE: $BATCH_DRAIN_SIZE"
  echo "ITERATIONS: $ITERATIONS"
  echo "FILE_LIMIT: $FILE_LIMIT"
  echo "MAIN_Q_CAPACITY=$MAIN_Q_CAPACITY"
  echo "AUX_Q_CAPACITY=$AUX_Q_CAPACITY"
else
  DECODER=$CPU_DECODER
fi

#these prameters work with both GPU and CPU decoder
echo "BEAM=$BEAM"
echo "LATTICE_BEAM=$LATTICE_BEAM"
echo "MAX_ACTIVE=$MAX_ACTIVE"
echo "DECODER=$DECODER"
echo "COMPUTE_CER=$COMPUTE_CER"
echo "OUTPUT_PATH=$OUTPUT_PATH"

#symlink files/folders expected to be in the current path
ln -sf $KALDI_ROOT/egs/wsj/s5/utils/
ln -sf ../path.sh

#compute output location
mkdir -p $OUTPUT_PATH
#compute which unique run this is.  This increments everytime this script is ran.
RUN=`ls -1 $OUTPUT_PATH | wc -l`

#unique output path
RESULT_PATH="$OUTPUT_PATH/$RUN"
mkdir -p $RESULT_PATH

WAVSCP=$DATASET/wav.scp

if [ -f $WAVSCP ]; then
  echo "Found wav.scp file using that"
else
  echo "No wav.scp file found.  Creating one now."

  #find all wavefiles in dataset
  WAVES=`find $DATASET -name *.wav`
  
  #make a new dataset directory and create wav.scp file
  DATASET=$RESULT_PATH/dataset
  
  mkdir -p $DATASET
  WAVSCP=$DATASET/wav.scp

  for file in $WAVES; do
    dir=`dirname $file`
    #translate / in paths to _
    dir=`echo $dir | tr "/" "_"`
    key=`basename $file .wav`

    key=${dir}_$key
    #remove .wav
    echo "$key $file" >> $WAVSCP
    echo "$key $key" >> $DATASET/utt2spk
  done
fi

echo "Computing segments"
#compute full audio segments for each file in wav.scp located at ${DATASET}
#this creates a segment file with the full length of audio to pass into get_uniform_subsegments
$KALDI_ROOT/egs/wsj/s5/utils/data/get_segments_for_data.sh $DATASET > $RESULT_PATH/full_segments

#compute subsegments
#this creates a list of subsegments with a fixed size.  The output is a subsegment file.
$KALDI_ROOT/egs/wsj/s5/utils/data/get_uniform_subsegments.py --overlap-duration 0.5 --max-segment-duration $SEGMENT_SIZE $RESULT_PATH/full_segments > $RESULT_PATH/subsegments

#compute segments
#this translates the subsegment file into a segment file that we can then decode on
$KALDI_ROOT/egs/wsj/s5/utils/data/subsegment_data_dir.sh $DATASET $RESULT_PATH/subsegments $RESULT_PATH

# copy vocabulary locally as lowercase (see below caveat for comment on this)
cat $MODEL_PATH/words.txt | tr '[:upper:]' '[:lower:]' > $RESULT_PATH/words.txt

#if transcript exists copy it to the result path and clean it up
if [ -f  $DATASET/text ]; then
  echo "Generating new reference transcripts for model and dataset..."
  #Lower all except the first field (file name)
  cat $DATASET/text | awk '{printf "%s ",$1; for(i=2;i<=NF;i++) {printf "%s ",tolower($i)} printf "\n"}' > $RESULT_PATH/gold_text
  if grep -q "<unk>" $RESULT_PATH/words.txt; then
     oovtok="--map-oov $(cat $RESULT_PATH/words.txt | grep "<unk>" | awk '{print $2}')"
  fi

  $KALDI_ROOT/egs/wsj/s5/utils/sym2int.pl $oovtok -f 2- $RESULT_PATH/words.txt $RESULT_PATH/gold_text > $RESULT_PATH/gold_text_ints 2> /dev/null
fi

#extract segments into an archive
echo "Extracting segments"
$KALDI_ROOT/src/featbin/extract-segments scp:$DATASET/wav.scp $RESULT_PATH/segments ark,scp:$RESULT_PATH/segments.ark,$RESULT_PATH/segments.scp
WAVIN="ark:$RESULT_PATH/segments.ark"

CUDAFLAGS=""
CPUFLAGS=""

if [ $USE_GPU -eq 1 ]; then
  NUM_CHANNELS=$(($MAX_BATCH_SIZE + $MAX_BATCH_SIZE/2))
  #Set CUDA decoder specific flags
  CUDAFLAGS="--gpu-feature-extract=$GPU_FEATURE --num-channels=$NUM_CHANNELS --cuda-use-tensor-cores=true --iterations=$ITERATIONS --main-q-capacity=$MAIN_Q_CAPACITY --aux-q-capacity=$AUX_Q_CAPACITY --cuda-memory-proportion=.5 --max-batch-size=$MAX_BATCH_SIZE --cuda-control-threads=$GPU_THREADS --batch-drain-size=$BATCH_DRAIN_SIZE --cuda-worker-threads=$WORKER_THREADS  --file-limit=$FILE_LIMIT --cuda-decoder-copy-threads=$COPY_THREADS"
  SPK2UTT=""
else
  SPK2UTT=ark:$RESULT_PATH/spk2utt.ark
  cat $RESULT_PATH/segments.scp | cut -f 1 -d " " > $RESULT_PATH/spk
  paste -d " " $RESULT_PATH/spk $RESULT_PATH/spk > $RESULT_PATH/spk2utt.ark
  CPUFLAGS="--online=false"
fi

#Set Generic flags
FLAGS="--frame-subsampling-factor=$FRAME_SUBSAMPLING_FACTOR --frames-per-chunk=$FRAMES_PER_CHUNK --max-mem=100000000 --beam=$BEAM --lattice-beam=$LATTICE_BEAM --acoustic-scale=1.0 --determinize-lattice=true --max-active=$MAX_ACTIVE"

PIDS=""
echo "Launching processes in parallel"
for (( d = 0 ; d < $NUM_PROCESSES ; d++ )); do
  LOCAL_RESULT_PATH=$RESULT_PATH/$d
  mkdir -p $LOCAL_RESULT_PATH
  export CUDA_VISIBLE_DEVICES=$d
  run_benchmark &
  PIDS+="$! "
done

FAIL=0
d=0
for pid in $PIDS; do
  if ! wait $pid; then
    echo "Process $d FAILED with error.  Output below:"
    cat $RESULT_PATH/$d/output.log
    FAIL=1
  fi 
  d=$((d+1))  
done

if [ $FAIL -eq 1 ]; then
  echo "Some tests FAILED"
  exit 1
else
  echo "All tests PASSED"
fi

TOTAL_RTF=0
for (( d = 0 ; d < $NUM_PROCESSES ; d++ )); do
  LOCAL_RESULT_PATH=$RESULT_PATH/$d
  echo "Process $d:"
  cat $LOCAL_RESULT_PATH/rtf
  cat $LOCAL_RESULT_PATH/wer

  if [ $USE_GPU -eq 1 ]; then
    RTF=`cat $LOCAL_RESULT_PATH/rtf | grep Aggregate | tail -n 1 | tr -s " " | cut -d " " -f 10`
  else
    RTF=`cat $LOCAL_RESULT_PATH/output.log| grep real-time | cut -d " " -f 12` 
    #CPU decoder returns the inverse
    RTF=`echo "1.0 / $RTF" | bc -l`
  fi
  WER=`cat $LOCAL_RESULT_PATH/wer  | grep WER | cut -d " " -f 4`
  TOTAL_RTF=`echo "$RTF + ${TOTAL_RTF}" | bc`

  if [ ! -z "$EXPECTED_WER" ]; then
    PASS=`echo "$WER <= $EXPECTED_WER" | bc`
    if [ $PASS -ne "1" ]; then
      echo "              Error:  WER rate ($WER) greater than  $EXPECTED_WER"
      FAIL=1
    fi
  fi
  if [ ! -z "$EXPECTED_PERF" ]; then
    PASS=`echo "$RTF >= $EXPECTED_PERF" | bc`
    if [ $PASS -ne "1" ]; then
      echo "              Error:  PERF ($RTF) less than than  $EXPECTED_PERF"
      FAIL=1
    fi
  fi
done
AVERAGE_RTF=`echo "scale=4; ${TOTAL_RTF} / $NUM_PROCESSES" | bc -l`
echo "Total RTF: ${TOTAL_RTF} Average RTF: $AVERAGE_RTF"

if [ $FAIL -eq 1 ]; then
  echo "Expected WER or PERF test failure.";
  exit 1
else
  echo "All WER and PERF tests passed."
fi
