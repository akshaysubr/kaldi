#!/bin/bash

mkdir -p /workspace/models
mkdir -p /workspace/datasets
mkdir -p /workspace/data

#wget http://sqrl/datasets/kaldi/swbd-eng.tar.gz
tar -xzf swbd-eng.tar.gz

ln -srf swbd-eng/model/ /workspace/models/swbd-eng
ln -srf swbd-eng/dataset/ /workspace/datasets/swbd-eng
ln -srf swbd-eng/data/ /workspace/data/swbd-eng

ln -sf ../run_benchmark.sh
