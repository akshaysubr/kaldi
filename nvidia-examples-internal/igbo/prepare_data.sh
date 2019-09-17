#!/bin/bash

wget http://sqrl/datasets/kaldi/igbo.tar.gz

tar -xzf igbo.tar.gz

mkdir -p /workspace/data
mkdir -p /workspace/datasets
mkdir -p /workspace/models

mv igbo/data /workspace/data/igbo
mv igbo/dataset /workspace/datasets/igbo
mv igbo/model /workspace/models/igbo

ln -sf ../run_benchmark.sh
