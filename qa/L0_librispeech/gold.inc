#!/bin/bash

declare -A GOLD_WER
declare -A GOLD_PERF

GOLD_WER=( 
  ["test_clean"]="5.56" 
  ["test_other"]="14.03" 
)

GOLD_PERF=(
  ["T4x2_test_clean"]="0"
  ["T4x2_test_other"]="0"
  ["P40x2_test_clean"]="0"
  ["P40x2_test_other"]="0"
  ["P100x4_test_clean"]="0"
  ["P100x4_test_other"]="0"
  ["V100x2_test_clean"]="0"
  ["V100x2_test_other"]="0"
  ["V100x8_test_clean"]="2300"
  ["V100x8_test_other"]="1700"
)
