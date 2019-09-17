#!/bin/bash

declare -A GOLD_WER
declare -A GOLD_PERF

GOLD_WER=( 
  ["igbo"]="0.0" 
)

GOLD_PERF=(
  ["T4x2_igbo"]="0"
  ["P40x2_igbo"]="0"
  ["P100x4_igbo"]="0"
  ["V100x8_igbo"]="200"
)
