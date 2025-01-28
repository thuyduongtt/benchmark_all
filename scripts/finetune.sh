#!/bin/bash

DS_VERSION="unbalanced"
#DS_DIR="../dataset/${DS_VERSION}"
DS_DIR="/fs/scratch/rb_bd_dlp_rng_dl01_cr_ICT_employees/trh7rng/dataset/${DS_VERSION}"
CONFIG_FILE=""

case $1 in
  1)
    CONFIG_FILE="qwen2.5-vl.yaml"
    ;;

esac

source activate finetune

# Step 1: Preparee the dataset
python -m finetune.prepare_dataset --ds_dir $DS_DIR

# Step 2: Start finetuning
#axolotl train finetune/${CONFIG_FILE}
#accelerate launch -m axolotl.cli.train finetune/${CONFIG_FILE}

#cp llama-factory/reasonvqa.json LLaMA-Factory/data/reasonvqa.json
#cd LLaMA-Factory
#llamafactory-cli train ../finetune/llama-factory_qwen2-vl.yaml