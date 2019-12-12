#!/bin/bash

mkdir -p /workspace/models
mkdir -p /workspace/datasets
mkdir -p /workspace/data

wget http://sqrl/datasets/kaldi/georgian.tar.gz
tar -xzf georgian.tar.gz

ln -srf georgian/model/ /workspace/models/georgian
ln -srf georgian/dataset/ /workspace/datasets/georgian
ln -srf georgian/data/ /workspace/data/georgian

ln -sf ../run_benchmark.sh
