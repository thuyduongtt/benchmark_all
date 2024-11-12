#!/bin/bash

LIMIT=20000
DS_NAME="ReasonVQA"
DS_VERSION="unbalanced"

DS_DIR="../dataset/${DS_VERSION}"
MODEL_TYPE=""

case $1 in
# ======================================== mPLUGOwl3
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
# ======================================== Idefics2
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
# ======================================== LLaVA
  17)
    CONDA_ENV="llava"
    MODEL_NAME="llava"
    START=0
    MULTICHOICE=true
    ;;
  18)
    CONDA_ENV="llava"
    MODEL_NAME="llava"
    START=20000
    MULTICHOICE=true
    ;;
  19)
    CONDA_ENV="llava"
    MODEL_NAME="llava"
    START=40000
    MULTICHOICE=true
    ;;
  20)
    CONDA_ENV="llava"
    MODEL_NAME="llava"
    START=60000
    MULTICHOICE=true
    ;;
  21)
    CONDA_ENV="llava"
    MODEL_NAME="llava"
    START=0
    MULTICHOICE=false
    ;;
  22)
    CONDA_ENV="llava"
    MODEL_NAME="llava"
    START=20000
    MULTICHOICE=false
    ;;
  23)
    CONDA_ENV="llava"
    MODEL_NAME="llava"
    START=40000
    MULTICHOICE=false
    ;;
  24)
    CONDA_ENV="llava"
    MODEL_NAME="llava"
    START=60000
    MULTICHOICE=false
    ;;
# ======================================== LLaVA-NEXT
  25)
    CONDA_ENV="llava_next"
    MODEL_NAME="llava_next_stronger"
    START=0
    MULTICHOICE=true
    ;;
  26)
    CONDA_ENV="llava_next"
    MODEL_NAME="llava_next_stronger"
    START=20000
    MULTICHOICE=true
    ;;
  27)
    CONDA_ENV="llava_next"
    MODEL_NAME="llava_next_stronger"
    START=40000
    MULTICHOICE=true
    ;;
  28)
    CONDA_ENV="llava_next"
    MODEL_NAME="llava_next_stronger"
    START=60000
    MULTICHOICE=true
    ;;
  29)
    CONDA_ENV="llava_next"
    MODEL_NAME="llava_next_stronger"
    START=0
    MULTICHOICE=false
    ;;
  30)
    CONDA_ENV="llava_next"
    MODEL_NAME="llava_next_stronger"
    START=20000
    MULTICHOICE=false
    ;;
  31)
    CONDA_ENV="llava_next"
    MODEL_NAME="llava_next_stronger"
    START=40000
    MULTICHOICE=false
    ;;
  32)
    CONDA_ENV="llava_next"
    MODEL_NAME="llava_next_stronger"
    START=60000
    MULTICHOICE=false
    ;;
# ======================================== Mantis-SIGLIP
  33)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    START=0
    MULTICHOICE=true
    ;;
  34)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    START=20000
    MULTICHOICE=true
    ;;
  35)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    START=40000
    MULTICHOICE=true
    ;;
  36)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    START=60000
    MULTICHOICE=true
    ;;
  37)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    START=0
    MULTICHOICE=false
    ;;
  38)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    START=20000
    MULTICHOICE=false
    ;;
  39)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    START=40000
    MULTICHOICE=false
    ;;
  40)
    CONDA_ENV="mantis_siglip"
    MODEL_NAME="mantis_siglip"
    START=60000
    MULTICHOICE=false
    ;;
# ======================================== Mantis-Idefics2
  41)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    START=0
    MULTICHOICE=true
    ;;
  42)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    START=20000
    MULTICHOICE=true
    ;;
  43)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    START=40000
    MULTICHOICE=true
    ;;
  44)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    START=60000
    MULTICHOICE=true
    ;;
  45)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    START=0
    MULTICHOICE=false
    ;;
  46)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    START=20000
    MULTICHOICE=false
    ;;
  47)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    START=40000
    MULTICHOICE=false
    ;;
  48)
    CONDA_ENV="owl3"
    MODEL_NAME="mantis_idefics2"
    START=60000
    MULTICHOICE=false
    ;;

esac


source activate $CONDA_ENV

OUTPUT_NAME=${MODEL_NAME}_${MODEL_TYPE}_${DS_NAME}_${DS_VERSION}_${START}

if [ "$MULTICHOICE" = true ] ; then
  python -m benchmark.start \
   --model_name $MODEL_NAME \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --output_dir_name output_mc_${OUTPUT_NAME} \
   --start_at $START \
   --limit $LIMIT \
   --multichoice
else
  python -m benchmark.start \
   --model_name $MODEL_NAME \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --output_dir_name output_${OUTPUT_NAME} \
   --start_at $START \
   --limit $LIMIT
fi







case $1 in
  1)
    python compute_score.py --model blip2 --ds unbalanced
    ;;
  2)
    python compute_score.py --model blip2 --ds balanced_10
    ;;
  3)
    python compute_score.py --model kosmos --ds unbalanced
    ;;
  4)
    python compute_score.py --model kosmos --ds balanced_10
    ;;
  5)
    python compute_score.py --model lavis --ds unbalanced
    ;;
  6)
    python compute_score.py --model lavis --ds balanced_10
    ;;
  7)
    python compute_score.py --model pretrain_opt6.7b --ds unbalanced
    ;;
  8)
    python compute_score.py --model pretrain_opt6.7b --ds balanced_10
    ;;
  9)
    python compute_score.py --model instructBLIP_flant --ds unbalanced
    ;;
  10)
    python compute_score.py --model instructBLIP_flant --ds balanced_10
    ;;
  11)
    python compute_score.py --model mPLUGOwl2 --ds unbalanced
    ;;
  12)
    python compute_score.py --model mPLUGOwl2 --ds VQAv2
    ;;
  13)
    python compute_score.py --model mPLUGOwl2 --ds OKVQA
    ;;
  14)
    python compute_score.py --model instructBLIP_flant --ds VQAv2
    ;;
  15)
    python compute_score.py --model instructBLIP_flant --ds OKVQA
    ;;
esac