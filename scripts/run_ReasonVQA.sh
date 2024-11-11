#!/bin/bash

LIMIT=20000
DS_NAME="ReasonVQA"
DS_VERSION="unbalanced"

DS_DIR="../dataset/${DS_VERSION}"

case $1 in
  1)
    MODEL_NAME="mPLUGOwl3"
    START=0
    MULTICHOICE=true
    ;;
  2)
    MODEL_NAME="mPLUGOwl3"
    START=20000
    MULTICHOICE=true
    ;;
  3)
    MODEL_NAME="mPLUGOwl3"
    START=40000
    MULTICHOICE=true
    ;;
  4)
    MODEL_NAME="mPLUGOwl3"
    START=60000
    MULTICHOICE=true
    ;;
  5)
    MODEL_NAME="mPLUGOwl3"
    START=0
    MULTICHOICE=false
    ;;
  6)
    MODEL_NAME="mPLUGOwl3"
    START=20000
    MULTICHOICE=false
    ;;
  7)
    MODEL_NAME="mPLUGOwl3"
    START=40000
    MULTICHOICE=false
    ;;
  8)
    MODEL_NAME="mPLUGOwl3"
    START=60000
    MULTICHOICE=false
    ;;
esac

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
