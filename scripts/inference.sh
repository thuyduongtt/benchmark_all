#!/bin/bash


DS_PATH="benchmark/dataset.json"

case $1 in
# ======================================== mPLUGOwl3
  1)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    ;;
  2)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    ;;

esac


source activate $CONDA_ENV

OUTPUT_NAME=${MODEL_NAME}

python -m benchmark.start \
  --ds_path $DS_PATH \
  --model_name $MODEL_NAME \
  --output_name inference_${OUTPUT_NAME}
