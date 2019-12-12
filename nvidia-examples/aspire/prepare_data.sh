#!/bin/bash

model=aspire

data=${1:-/workspace/data/}
datasets=/workspace/datasets/
models=/workspace/models/

# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11
mfccdir=mfcc

mkdir -p $data
mkdir -p $models/$model
mkdir -p $datasets/$model

pushd /opt/kaldi/egs/librispeech/s5

. ./cmd.sh
. ./path.sh
. parse_options.sh

# you might not want to do this for interactive shells.
set -e

if [[ "$SKIP_DATA_DOWNLOAD" -ne "1" ]]; then
  echo ----------- Fetching dataset -----------
  # download the data.  
  for part in test-clean test-other; do
    local/download_and_untar.sh $data $data_url $part
  done
  cp -R $data/LibriSpeech/ $data/aspire
fi

# format the data as Kaldi data directories
echo ----------- Preprocessing dataset -----------
for part in test-clean test-other; do
  # use underscore-separated names in data directories.
  local/data_prep.sh $data/$model/$part $datasets/$model/$(echo $part | sed s/-/_/g)
  # convert the manifests
  pushd $datasets/$model/$(echo $part | sed s/-/_/g); 
  mv wav.scp wav16k.scp
  cat wav16k.scp | awk '{print $1" "$6}' | sed 's/\.flac/\.wav/g' > wav8k.scp
  cp wav8k.scp wav.scp
  popd
done

if [[ "$SKIP_FLAC2WAV" -ne "1" ]]; then
  # Convert flac files to wavs
  for flac in $(find $data/$model -name "*.flac"); do
     wav=$(echo $flac | sed 's/flac/wav/g')
     sox $flac -r 8000 -b 16 $wav
  done

  echo "Converted flac to wav."
fi

popd 
 
CWD=$PWD

if [[ "$SKIP_MODEL_DOWNLOAD" -ne "1" ]]; then
  echo ----------- Fetching trained model -----------

  MODEL_SRC=$KALDI_ROOT/egs/aspire/s5

  pushd $models/$model

  #create symlinks to model recipie
  for file in $MODEL_SRC/*; do
    ln -sf $file
  done

  export PATH=$PATH:./utils/:$KALDI_ROOT/src/bin/
  #donwload and unpack model
  wget https://kaldi-asr.org/models/1/0001_aspire_chain_model_with_hclg.tar.bz2 
  tar --no-same-owner -xjf 0001_aspire_chain_model_with_hclg.tar.bz2

  #we will overwrite path script pulled from the MODEL directory with our own
  rm -f path.sh

  #symlink the path.sh file
  ln -s $CWD/../path.sh ./path.sh

  #model prep
  time steps/online/nnet3/prepare_online_decoding.sh   --mfcc-config conf/mfcc_hires.conf data/lang_chain exp/nnet3/extractor exp/chain/tdnn_7b exp/tdnn_7b_chain_online
  #this generates the fst but we downloaded it precompiled to save time
  #time utils/mkgraph.sh --self-loop-scale 1.0 data/lang_pp_test exp/tdnn_7b_chain_online exp/tdnn_7b_chain_online/graph_pp
  time utils/build_const_arpa_lm.sh data/local/lm/4gram-mincount/lm_unpruned.gz data/lang_pp_test data/lang_pp_test_fg

  #create model bits in expected location
  ln -sf `pwd`/exp/tdnn_7b_chain_online/graph_pp/HCLG.fst 
  ln -sf `pwd`/exp/tdnn_7b_chain_online/graph_pp/words.txt
  ln -sf `pwd`/exp/tdnn_7b_chain_online/final.mdl 
  ln -sf `pwd`/exp/tdnn_7b_chain_online/conf/
  ln -sf `pwd`/exp/tdnn_7b_chain_online/graph_pp/phones

  #filter out unsupported options
  cp ./conf/online.conf ./conf/online.conf.backup
  cat ./conf/online.conf.backup | grep -v silence > ./conf/online.conf
  
  popd

fi

ln -sf ../run_benchmark.sh
