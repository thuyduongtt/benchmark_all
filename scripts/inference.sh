#!/bin/bash


DS_PATH="dataset.json"
MODEL_TYPE=""

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

OUTPUT_NAME=${MODEL_NAME}_${MODEL_TYPE}

python -m benchmark.start \
  --ds_path $DS_PATH \
  --model_name $MODEL_NAME \
  --model_type $MODEL_TYPE \
  --output_name inference_${OUTPUT_NAME}
