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

popd >&/dev/null

if [[ "$SKIP_MODEL_DOWNLOAD" -ne "1" ]]; then
  echo ----------- Fetching trained model -----------
  pushd $models >&/dev/null
  wget http://sqrl/datasets/speech/models/aspire-trained.tar.gz -O aspire-trained.tgz
  tar -xzf aspire-trained.tgz
  popd >&/dev/null
fi

ln -s ../run_benchmark.sh
