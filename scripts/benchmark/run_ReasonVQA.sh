#!/bin/bash

# Ensure input is a valid number
if [[ ! "$1" =~ ^[0-9]+$ ]] || [[ "$1" -lt 1 ]]; then
  echo "Usage: $0 <positive integer>"
  exit 1
fi

# List of models and conda env
# the model is selected by argument $1
# dataset is divided into 4 parts to run benckmark in 2 scenarios: multi-choice and open-ended
# each model corresponds to 8 indices, first 4 indices are multi-choice, the last 4 indices are open-ended
MODELS=(
"mPLUGOwl3 owl3"  # 1-4, 5-8
"idefics2 owl3"  # 9-12, 13-16
"llava llava"  # 17-20, 21-24
"llava_next_stronger llava_next"  # 25-28, 29-32
"mantis_siglip mantis_siglip"  # 33-36, 37-40
"mantis_idefics2 owl3"  # 41-44, 45-48
"llava_ov llava_ov"  # 49-52, 53-56
"qwen25 qwen"  # 57-60, 61-64
"gpt openai"  # 65-68, 69-72
"qwen2 qwen"  # 73-76, 77-80
"qwen2finetuned qwen"  # 81-84, 85-88
"paligemma2 owl3"  # 89-92, 93-96
"paligemma2mix owl3"  # 97-100, 101-104
"paligemma2mix3b owl3"  #105-108, 109-112
"smolvlm owl3"  #113-116, 117-120
"paligemma2mix_ft owl3"  #121-124, 125-128
"paligemma2mix3b_ft owl3"  #129-132, 133-136
)


SPLIT="test"

if [[ $SPLIT == "train" ]]; then
  LIMIT=13000  # train: 51,829 / 4 ≈ 13000
else
  LIMIT=6000  # test: 22,832 / 4 ≈ 6,000
fi

N_PART=4  # divide the dataset into parts, each contains $LIMIT samples
N_PART_SCENARIO=$(( $N_PART * 2 ))  # two scenarios: multi-choice & open-ended

# Calculate which model to use
MODEL_INDEX=$(( ($1 - 1) / $N_PART_SCENARIO ))  # Each model has 8 cases
MODEL_ENTRY=(${MODELS[$MODEL_INDEX]})  # Split the string into an array
MODEL_NAME="${MODEL_ENTRY[0]}"
CONDA_ENV="${MODEL_ENTRY[1]}"

# Calculate START
START=$(( ($1 - 1) % $N_PART * $LIMIT ))

# Determine MULTICHOICE value
if (( ($1 - 1) % $N_PART_SCENARIO < $N_PART )); then
  MULTICHOICE=true
else
  MULTICHOICE=false
fi

echo "MODEL_NAME=$MODEL_NAME"
echo "CONDA_ENV=$CONDA_ENV"
echo "START=$START"
echo "MULTICHOICE=$MULTICHOICE"
echo "LIMIT=$LIMIT"

# exit 0

source activate $CONDA_ENV


DS_NAME="ReasonVQA"
DS_VERSION="unbalanced"

DS_DIR="../dataset/${DS_VERSION}"
MODEL_TYPE=""
VISUAL_DISABLED=false


OUTPUT_NAME=${MODEL_NAME}_${MODEL_TYPE}_${DS_NAME}_${DS_VERSION}_${SPLIT}_${START}

if [ "$MULTICHOICE" = true ] ; then
  if [ "$VISUAL_DISABLED" = false ] ; then
   python -m benchmark.start \
    --model_name $MODEL_NAME \
    --ds_name $DS_NAME \
    --ds_dir $DS_DIR \
    --output_dir_name output_mc_${OUTPUT_NAME} \
    --start_at $START \
    --limit $LIMIT \
    --split $SPLIT \
    --multichoice
  else
   python -m benchmark.start \
     --model_name $MODEL_NAME \
     --ds_name $DS_NAME \
     --ds_dir $DS_DIR \
     --output_dir_name output_mc_${OUTPUT_NAME} \
     --start_at $START \
     --limit $LIMIT \
     --split $SPLIT \
     --multichoice \
     --visual_disabled
  fi
else
  if [ "$VISUAL_DISABLED" = false ] ; then
    python -m benchmark.start \
     --model_name $MODEL_NAME \
     --ds_name $DS_NAME \
     --ds_dir $DS_DIR \
     --output_dir_name output_${OUTPUT_NAME} \
     --start_at $START \
     --limit $LIMIT \
     --split $SPLIT
  else
   python -m benchmark.start \
    --model_name $MODEL_NAME \
    --ds_name $DS_NAME \
    --ds_dir $DS_DIR \
    --output_dir_name output_${OUTPUT_NAME} \
    --start_at $START \
    --limit $LIMIT \
    --split $SPLIT \
    --visual_disabled
  fi
fi
