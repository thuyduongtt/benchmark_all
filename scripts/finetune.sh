#!/bin/bash

DS_VERSION="unbalanced"
DS_DIR="../dataset/${DS_VERSION}"
CONFIG_FILE=""

case $1 in
  1)
    CONFIG_FILE="qwen2.5-vl.yaml"
    ;;

esac

source activate finetune

# Step 1: Preparee the dataset
#python -m finetune.prepare_dataset --ds_dir $DS_DIR

# Step 2: Start finetuning
axolotl train -c finetune/${CONFIG_FILE}