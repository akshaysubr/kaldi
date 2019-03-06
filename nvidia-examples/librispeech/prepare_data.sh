#!/bin/bash

model=LibriSpeech

data=/workspace/data/
datasets=/workspace/datasets/
models=/workspace/models/

# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11
mfccdir=mfcc

mkdir -p $data/$model
mkdir -p $models/$model
mkdir -p $datasets/$model

pushd /opt/kaldi/egs/librispeech/s5 >/dev/null

. ./cmd.sh
. ./path.sh
. parse_options.sh

# you might not want to do this for interactive shells.
set -e

echo ----------- Fetching dataset -----------

# download the data.  Note: we're using the 100 hour setup for
# now; later in the script we'll download more and use it to train neural
# nets.
for part in test-clean test-other; do
  local/download_and_untar.sh $data $data_url $part
done


echo ----------- Preprocessing dataset -----------

# format the data as Kaldi data directories
for part in test-clean test-other; do
  # use underscore-separated names in data directories.
  local/data_prep.sh $data/$model/$part $datasets/$model/$(echo $part | sed s/-/_/g)
  # convert the manifests
  pushd $datasets/$model/$(echo $part | sed s/-/_/g); (cat wav.scp | awk '{print $1" "$6}' | sed 's/\.flac/\.wav/g' > wav_conv.scp); popd
done

# Convert flag files to wavs
for flac in $(find $data/$model -name "*.flac"); do
   wav=$(echo $flac | sed 's/flac/wav/g')
   sox $flac -r 16000 -b 16 $wav
done

echo "Converted flac to wav."

popd >/dev/null

#TODO get from model repository 
echo ----------- Fetching trained model -----------
pushd $models >/dev/null
wget http://sqrl/dldata/speech/LibriSpeech-trained.tgz
tar -xzf LibriSpeech-trained.tgz -C $model
popd >/dev/null
