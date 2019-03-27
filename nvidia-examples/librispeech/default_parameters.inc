#!/bin/bash

GPU_THREADS=${GPU_THREADS:-3}
MODEL_PATH=${MODEL_PATH:-"/workspace/models/LibriSpeech/"}
DATASET_PATH=${DATASET_PATH:-"/workspace/datasets/LibriSpeech/"}
DATASETS=${DATASETS:-"test_clean test_other"}
MAIN_Q_CAPACITY=${MAIN_Q_CAPACITY:-30000}
AUX_Q_CAPACITY=${AUX_Q_CAPACITY:-400000}
