#!/bin/bash

LIMIT=20000
DS_NAME="ReasonVQA"
DS_VERSION="unbalanced"

DS_DIR="../dataset/${DS_VERSION}"
MODEL_TYPE=""

case $1 in
  1)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    START=0
    MULTICHOICE=true
    ;;
  2)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    START=20000
    MULTICHOICE=true
    ;;
  3)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    START=40000
    MULTICHOICE=true
    ;;
  4)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    START=60000
    MULTICHOICE=true
    ;;
  5)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    START=0
    MULTICHOICE=false
    ;;
  6)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    START=20000
    MULTICHOICE=false
    ;;
  7)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    START=40000
    MULTICHOICE=false
    ;;
  8)
    CONDA_ENV="owl3"
    MODEL_NAME="mPLUGOwl3"
    START=60000
    MULTICHOICE=false
    ;;

  9)
    CONDA_ENV="owl3"
    MODEL_NAME="idefics2"
    START=0
    MULTICHOICE=true
    ;;
  10)
    CONDA_ENV="owl3"
    MODEL_NAME="idefics2"
    START=20000
    MULTICHOICE=true
    ;;
  11)
    CONDA_ENV="owl3"
    MODEL_NAME="idefics2"
    START=40000
    MULTICHOICE=true
    ;;
  12)
    CONDA_ENV="owl3"
    MODEL_NAME="idefics2"
    START=60000
    MULTICHOICE=true
    ;;
  13)
    CONDA_ENV="owl3"
    MODEL_NAME="idefics2"
    START=0
    MULTICHOICE=false
    ;;
  14)
    CONDA_ENV="owl3"
    MODEL_NAME="idefics2"
    START=20000
    MULTICHOICE=false
    ;;
  15)
    CONDA_ENV="owl3"
    MODEL_NAME="idefics2"
    START=40000
    MULTICHOICE=false
    ;;
  16)
    CONDA_ENV="owl3"
    MODEL_NAME="idefics2"
    START=60000
    MULTICHOICE=false
    ;;

  17)
    CONDA_ENV="llava-next"
    MODEL_NAME="llava_next_stronger"
    START=0
    MULTICHOICE=true
    ;;
  18)
    CONDA_ENV="llava-next"
    MODEL_NAME="llava_next_stronger"
    START=20000
    MULTICHOICE=true
    ;;
  19)
    CONDA_ENV="llava-next"
    MODEL_NAME="llava_next_stronger"
    START=40000
    MULTICHOICE=true
    ;;
  20)
    CONDA_ENV="llava-next"
    MODEL_NAME="llava_next_stronger"
    START=60000
    MULTICHOICE=true
    ;;
  21)
    CONDA_ENV="llava-next"
    MODEL_NAME="llava_next_stronger"
    START=0
    MULTICHOICE=false
    ;;
  22)
    CONDA_ENV="llava-next"
    MODEL_NAME="llava_next_stronger"
    START=20000
    MULTICHOICE=false
    ;;
  23)
    CONDA_ENV="llava-next"
    MODEL_NAME="llava_next_stronger"
    START=40000
    MULTICHOICE=false
    ;;
  24)
    CONDA_ENV="llava-next"
    MODEL_NAME="llava_next_stronger"
    START=60000
    MULTICHOICE=false
    ;;
esac


source activate $CONDA_ENV

OUTPUT_NAME=${MODEL_NAME}_${MODEL_TYPE}_${DS_NAME}_${DS_VERSION}_${START}

if [ "$MULTICHOICE" = true ] ; then
  python start.py \
   --model_name $MODEL_NAME \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --output_dir_name output_mc_${OUTPUT_NAME} \
   --start_at $START \
   --limit $LIMIT \
   --multichoice
else
  python start.py \
   --model_name $MODEL_NAME \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --output_dir_name output_${OUTPUT_NAME} \
   --start_at $START \
   --limit $LIMIT
fi
