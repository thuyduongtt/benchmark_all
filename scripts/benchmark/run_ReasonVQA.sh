#!/bin/bash

# Ensure input is a valid number
if [[ ! "$1" =~ ^[0-9]+$ ]] || [[ "$1" -lt 1 ]]; then
  echo "Usage: $0 <positive integer>"
  exit 1
fi

# List of models and conda env
# the model is selected by argument $1
# for each model, first half is multi-choice, the second half is open-ended
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
)

# Define models that should be benchmarked on val set only
VAL_MODELS=("qwen2" "qwen2finetuned")

LIMIT=20000
N_PART=4  # divide the dataset into parts, each contains $LIMIT samples
N_PART_SCENARIO=$(( $N_PART * 2 ))  # two scenarios: multi-choice & open-ended

# Calculate which model to use
MODEL_INDEX=$(( ($1 - 1) / $N_PART_SCENARIO ))  # Each model has 8 cases
MODEL_ENTRY=(${MODELS[$MODEL_INDEX]})  # Split the string into an array
MODEL_NAME="${MODEL_ENTRY[0]}"
CONDA_ENV="${MODEL_ENTRY[1]}"

for SPECIAL_MODEL in "${VAL_MODELS[@]}"; do
  if [[ "$MODEL_NAME" == "$SPECIAL_MODEL" ]]; then
    LIMIT=5000
    break
  fi
done

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
