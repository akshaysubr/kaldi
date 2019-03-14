#!/bin/bash

declare -A EXPECTED_WER
declare -A EXPECTED_PERF

EXPECTED_WER=( 
  ["test_clean"]="20.94" 
  ["test_other"]="40.67" 
)

EXPECTED_PERF=(
  ["2xT4_test_clean"]="500"
  ["2xT4_test_other"]="400"
  ["2xP40_test_clean"]="500"
  ["2xP40_test_other"]="400"
  ["4xP100_test_clean"]="500"
  ["4xP100_test_other"]="400"
  ["4xV100_test_clean"]="0"
  ["4xV100_test_other"]="0"
  ["8xV100_test_clean"]="500"
  ["8xV100_test_other"]="250"
)
