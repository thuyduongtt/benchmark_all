#!/bin/bash

DS_VERSION="unbalanced"
#DS_DIR="../dataset/${DS_VERSION}"
DS_DIR="/fs/scratch/rb_bd_dlp_rng_dl01_cr_ICT_employees/trh7rng/dataset/${DS_VERSION}"

source activate finetune

# Step 1: Prepare the dataset
#python -m finetune.prepare_dataset --ds_dir $DS_DIR

# Step 2: Start finetuning
cp finetune_ds/reasonvqa_llama_factory_train.json LLaMA-Factory/data/reasonvqa.json
cd LLaMA-Factory
llamafactory-cli train ../finetune/LF_finetune_qwen2-vl.yaml

# Step 3: Merge LoRA
#cd LLaMA-Factory
#llamafactory-cli export ../finetune/LF_merge_qwen2-vl.yaml
