#!/bin/bash

declare -A GOLD_WER
declare -A GOLD_PERF

GOLD_WER=( 
  ["iv_en-US"]="0.0" 
)

GOLD_PERF=(
  ["T4x2_iv_en-US"]="0"
  ["P40x2_iv_en-US"]="0"
  ["P100x4_iv_en-US"]="0"
  ["V100x8_iv_en-US"]="200"
)
