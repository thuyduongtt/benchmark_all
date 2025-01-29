#!/bin/bash

LIMIT=30000
DS_NAME="VQAv2"
MULTICHOICE=false

DS_DIR="../dataset/${DS_NAME}"
IMG_DIR="../dataset/COCO/val2014"

case $1 in
  1)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    START=0
    ;;
  2)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    START=30000
    ;;
  3)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    START=60000
    ;;
  4)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    START=90000
    ;;
  5)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    START=120000
    ;;
  6)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    START=150000
    ;;
  7)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    START=180000
    ;;
  8)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    START=210000
    ;;

  9)
    CONDA_ENV="owl3"
    MODEL_NAME="idefics2"
    START=0
    ;;
  10)
    CONDA_ENV="owl3"
    MODEL_NAME="idefics2"
    START=30000
    ;;
  11)
    CONDA_ENV="owl3"
    MODEL_NAME="idefics2"
    START=60000
    ;;
  12)
    CONDA_ENV="owl3"
    MODEL_NAME="idefics2"
    START=90000
    ;;
  13)
    CONDA_ENV="owl3"
    MODEL_NAME="idefics2"
    START=120000
    ;;
  14)
    CONDA_ENV="owl3"
    MODEL_NAME="idefics2"
    START=150000
    ;;
  15)
    CONDA_ENV="owl3"
    MODEL_NAME="idefics2"
    START=180000
    ;;
  16)
    CONDA_ENV="owl3"
    MODEL_NAME="idefics2"
    START=210000
    ;;

  17)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    START=0
    ;;
  18)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    START=30000
    ;;
  19)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    START=60000
    ;;
  20)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    START=90000
    ;;
  21)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    START=120000
    ;;
  22)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    START=150000
    ;;
  23)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    START=180000
    ;;
  24)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    START=210000
    ;;

  25)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    START=0
    ;;
  26)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    START=30000
    ;;
  27)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    START=60000
    ;;
  28)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    START=90000
    ;;
  29)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    START=120000
    ;;
  30)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    START=150000
    ;;
  31)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    START=180000
    ;;
  32)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    START=210000
    ;;

  33)
    CONDA_ENV="llava_ov"
    MODEL_NAME="llava_ov"
    START=0
    ;;
  34)
    CONDA_ENV="llava_ov"
    MODEL_NAME="llava_ov"
    START=30000
    ;;
  35)
    CONDA_ENV="llava_ov"
    MODEL_NAME="llava_ov"
    START=60000
    ;;
  36)
    CONDA_ENV="llava_ov"
    MODEL_NAME="llava_ov"
    START=90000
    ;;
  37)
    CONDA_ENV="llava_ov"
    MODEL_NAME="llava_ov"
    START=120000
    ;;
  38)
    CONDA_ENV="llava_ov"
    MODEL_NAME="llava_ov"
    START=150000
    ;;
  39)
    CONDA_ENV="llava_ov"
    MODEL_NAME="llava_ov"
    START=180000
    ;;
  40)
    CONDA_ENV="llava_ov"
    MODEL_NAME="llava_ov"
    START=210000
    ;;

  41)
    CONDA_ENV="qwen"
    MODEL_NAME="qwen25"
    START=0
    ;;
  42)
    CONDA_ENV="qwen"
    MODEL_NAME="qwen25"
    START=30000
    ;;
  43)
    CONDA_ENV="qwen"
    MODEL_NAME="qwen25"
    START=60000
    ;;
  44)
    CONDA_ENV="qwen"
    MODEL_NAME="qwen25"
    START=90000
    ;;
  45)
    CONDA_ENV="qwen"
    MODEL_NAME="qwen25"
    START=120000
    ;;
  46)
    CONDA_ENV="qwen"
    MODEL_NAME="qwen25"
    START=150000
    ;;
  47)
    CONDA_ENV="qwen"
    MODEL_NAME="qwen25"
    START=180000
    ;;
  48)
    CONDA_ENV="qwen"
    MODEL_NAME="qwen25"
    START=210000
    ;;
esac

source activate $CONDA_ENV

OUTPUT_NAME=${MODEL_NAME}_${MODEL_TYPE}_${DS_NAME}_${START}

if [ "$MULTICHOICE" = true ] ; then
  python -m benchmark.start \
   --model_name $MODEL_NAME \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --img_dir $IMG_DIR \
   --output_dir_name output_mc_${OUTPUT_NAME} \
   --start_at $START \
   --limit $LIMIT \
   --multichoice
else
  python -m benchmark.start \
   --model_name $MODEL_NAME \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --img_dir $IMG_DIR \
   --output_dir_name output_${OUTPUT_NAME} \
   --start_at $START \
   --limit $LIMIT
fi