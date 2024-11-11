#!/bin/bash

DS_NAME="OKVQA"

case $1 in
  1)
    MULTICHOICE=true
    ;;
  2)
    MULTICHOICE=false
    ;;

DS_DIR="../dataset/${DS_NAME}"
IMG_DIR="../dataset/COCO/val2014"

if [ "$MULTICHOICE" = true ] ; then
  python start.py \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --img_dir $IMG_DIR \
   --output_dir_name output_mc_${DS_NAME} \
   --multichoice

else
  python start.py \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --img_dir $IMG_DIR \
   --output_dir_name output_${DS_NAME}
fi
