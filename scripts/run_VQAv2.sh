#!/bin/bash

LIMIT=30000
DS_NAME="VQAv2"
MULTICHOICE=true

DS_DIR="../dataset/${DS_NAME}"
IMG_DIR="../dataset/COCO/val2014"

case $1 in
  1)
    START=0
    ;;
  2)
    START=30000
    ;;
  3)
    START=60000
    ;;
  4)
    START=90000
    ;;
  5)
    START=120000
    ;;
  6)
    START=150000
    ;;
  7)
    START=180000
    ;;
  8)
    START=210000
    ;;
esac

if [ "$MULTICHOICE" = true ] ; then
  python start.py \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --img_dir $IMG_DIR \
   --output_dir_name output_mc_${DS_NAME}_${START} \
   --start_at $START \
   --limit $LIMIT \
   --multichoice

else
  python start.py \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --img_dir $IMG_DIR \
   --output_dir_name output_${DS_NAME}_${START} \
   --start_at $START \
   --limit $LIMIT
fi
