#!/bin/bash

GPU_THREADS=${GPU_THREADS:-2}
MODEL_PATH="/workspace/models/aspire/"
DATASET_PATH="/workspace/datasets/aspire/"
DATASETS="test_clean test_other"
MAIN_Q_CAPACITY=${MAIN_Q_CAPACITY:-40000}
AUX_Q_CAPACITY=${AUX_Q_CAPACITY:-400000}
