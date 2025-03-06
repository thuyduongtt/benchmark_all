#!/bin/bash

DS_VERSION="unbalanced"
DS_DIR="../dataset/${DS_VERSION}"
#DS_DIR="/fs/scratch/rb_bd_dlp_rng_dl01_cr_ICT_employees/trh7rng/dataset/${DS_VERSION}"

source activate finetune


python -m finetune.PaliGemma2 --ds_dir $DS_DIR
