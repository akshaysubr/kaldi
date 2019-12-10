#!/bin/bash

declare -A GOLD_WER
declare -A GOLD_PERF

GOLD_WER=( 
  ["swbd-eng"]="28.5" 
)

GOLD_PERF=(
  ["T4x2_swbd-eng"]="0"
  ["P40x2_swbd-eng"]="0"
  ["P100x4_swbd-eng"]="0"
  ["V100x8_swbd-eng"]="0"
)
