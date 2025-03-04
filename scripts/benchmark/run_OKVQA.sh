#!/bin/bash

DS_NAME="OKVQA"
MULTICHOICE=false

DS_DIR="../dataset/${DS_NAME}"
IMG_DIR="../dataset/COCO/val2014"

case $1 in
  1)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    ;;
  2)
    CONDA_ENV="owl3"
    MODEL_NAME="idefics2"
    ;;
  3)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    ;;
  4)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    ;;
  5)
    CONDA_ENV="lavis"
    MODEL_NAME="blip2_t5"
    ;;
  6)
    CONDA_ENV="llava_ov"
    MODEL_NAME="llava_ov"
    ;;
  7)
    CONDA_ENV="qwen"
    MODEL_NAME="qwen2"
    ;;
  8)
    CONDA_ENV="qwen"
    MODEL_NAME="qwen25"
    ;;
  9)
    CONDA_ENV="qwen"
    MODEL_NAME="qwen2finetuned"
    ;;
  10)
    CONDA_ENV="openai"
    MODEL_NAME="gpt"
    ;;
  11)
    CONDA_ENV="owl3"
    MODEL_NAME="paligemma2"
    ;;
  12)
    CONDA_ENV="owl3"
    MODEL_NAME="smolvlm"
    ;;
esac

source activate $CONDA_ENV

OUTPUT_NAME=${MODEL_NAME}_${MODEL_TYPE}_${DS_NAME}

if [ "$MULTICHOICE" = true ] ; then
  python -m benchmark.start \
   --model_name $MODEL_NAME \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --img_dir $IMG_DIR \
   --output_dir_name output_mc_${OUTPUT_NAME} \
   --multichoice
else
  python -m benchmark.start \
   --model_name $MODEL_NAME \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --img_dir $IMG_DIR \
   --output_dir_name output_${OUTPUT_NAME}
fi