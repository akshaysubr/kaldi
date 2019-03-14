#!/bin/bash

declare -A EXPECTED_WER
declare -A EXPECTED_PERF

EXPECTED_WER=( 
  ["test_clean"]="20.99" 
  ["test_other"]="40.71" 
)

EXPECTED_PERF=(
  ["2xT4_test_clean"]="500"
  ["2xT4_test_other"]="400"
  ["2xP40_test_clean"]="400"
  ["2xP40_test_other"]="300"
  ["4xP100_test_clean"]="400"
  ["4xP100_test_other"]="300"
  ["4xV100_test_clean"]="0"
  ["4xV100_test_other"]="0"
  ["8xV100_test_clean"]="500"
  ["8xV100_test_other"]="250"
)
