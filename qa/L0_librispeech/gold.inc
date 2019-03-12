#!/bin/bash

declare -A EXPECTED_WER
declare -A EXPECTED_PERF

EXPECTED_WER=( 
  ["test_clean"]="5.55" 
  ["test_other"]="14.02" 
)

EXPECTED_PERF=(
  ["2xT4_test_clean"]="1300"
  ["2xT4_test_other"]="1100"
  ["2xP40_test_clean"]="1300"
  ["2xP40_test_other"]="1150"
  ["4xP100_test_clean"]="1400"
  ["4xP100_test_other"]="1200"
  ["4xV100_test_clean"]="1400"
  ["4xV100_test_other"]="1200"
  ["8xV100_test_clean"]="800"
  ["8xV100_test_other"]="700"
)
