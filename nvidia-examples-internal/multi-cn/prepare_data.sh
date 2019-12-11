#!/bin/bash

mkdir -p /workspace/models
mkdir -p /workspace/datasets
mkdir -p /workspace/data

rm -Rf /multi-cn/model/multi-cn
rm -Rf /multi-cn/data/multi-cn
rm -Rf /multi-cn/dataset/multi-cn

wget http://sqrl/datasets/kaldi/multi-cn.tar.gz
tar -xzf multi-cn.tar.gz

ln -srf multi-cn/model/ /workspace/models/multi-cn
ln -srf multi-cn/dataset/ /workspace/datasets/multi-cn
ln -srf multi-cn/data/ /workspace/data/multi-cn

ln -sf ../run_benchmark.sh
