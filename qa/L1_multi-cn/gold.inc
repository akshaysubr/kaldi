#!/bin/bash

declare -A GOLD_WER
declare -A GOLD_PERF

GOLD_WER=( 
  ["multi-cn"]="10.7" 
)

GOLD_PERF=(
  ["T4x2_multi-cn"]="0"
  ["P40x2_multi-cn"]="0"
  ["P100x4_multi-cn"]="0"
  ["V100x8_multi-cn"]="0"
)
