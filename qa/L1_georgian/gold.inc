#!/bin/bash

declare -A GOLD_WER
declare -A GOLD_PERF

GOLD_WER=( 
  ["georgian"]="0.0" 
)

GOLD_PERF=(
  ["T4x2_georgian"]="0"
  ["P40x2_georgian"]="0"
  ["P100x4_georgian"]="0"
  ["V100x8_georgian"]="0"
)
