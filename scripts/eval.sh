#!/bin/bash


case $1 in
  1)
    RESULT_DIR="output_blip2_t5_instruct_flant5xxl_ReasonVQA_unbalanced"
    ;;
  2)
    RESULT_DIR="output_blip2_t5_pretrain_flant5xl_ReasonVQA_unbalanced"
    ;;
  3)
    RESULT_DIR="output_mantis_siglip__ReasonVQA_unbalanced"
    ;;
  4)
    RESULT_DIR="output_mPLUGOwl2__ReasonVQA_unbalanced"
    ;;
  5)
    RESULT_DIR="output_mPLUGOwl3__ReasonVQA_unbalanced"
    ;;
esac


source activate evaluation

python -m evaluation.compute_score --result_dir results/$RESULT_DIR

