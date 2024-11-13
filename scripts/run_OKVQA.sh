#!/bin/bash

DS_NAME="OKVQA"
MULTICHOICE=true

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
  2)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    ;;
  2)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    ;;

source activate $CONDA_ENV

OUTPUT_NAME=${MODEL_NAME}_${MODEL_TYPE}_${DS_NAME}_${DS_VERSION}_${START}

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