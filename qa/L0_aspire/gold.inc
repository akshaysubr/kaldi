#!/bin/bash

declare -A GOLD_WER
declare -A GOLD_PERF

GOLD_WER=( 
  ["test_clean"]="20.99" 
  ["test_other"]="40.79" 
)

GOLD_PERF=(
  ["T4x2_test_clean"]="0"
  ["T4x2_test_other"]="0"
  ["P40x2_test_clean"]="0"
  ["P40x2_test_other"]="0"
  ["P100x4_test_clean"]="0"
  ["P100x4_test_other"]="0"
  ["V100x4_test_clean"]="0"
  ["V100x4_test_other"]="0"
  ["V100x4_test_clean"]="400"
  ["V100x4_test_other"]="250"
)
