#!/bin/bash

CONFIG="./config.sh"

[[ ! -f "${CONFIG}" ]] && echo "${CONFIG} not found: Create ${CONFIG} from config.example.sh with your configurations." && exit 1

# Import configurations and export for script.
set -a
source ${CONFIG}
set +a

[[ -z "$CONVERTED_DIR" ]] && echo "CONVERTED_DIR not set in config.sh" && exit 1
[[ -z "$VENV" ]]     && echo "VENV not set in config.sh" && exit 1

MODEL_SIZE=<model size>
CONVERTED_LLAMA_MODEL=$CONVERTED_DIR/$MODEL_SIZE/state_dict.pth
TOKENIZER_PATH=$CONVERTED_DIR/tokenizer.model

MODEL_ID=<identifier for model>
CKPT_DIR=<directory with saved checkpoints>
CFG_PATH=$CKPT_DIR/config.json
CKPT_NUM=<checkpoint to evaluate>
CKPT_PATH=$CKPT_DIR/model-$CKPT_NUM.pt

source $VENV/bin/activate

DATA_DIR=<path to data dir>
OUTPUT_DIR=<path to store translations>

DATASETS=("flores" "wmt" "law" "medical" "tico" "chat_wmt")
declare -A DATASETS_LPS=(
    ["flores"]="de-en en-de fr-en en-fr nl-en en-nl pt-en en-pt ru-en en-ru zh-en en-zh"
    ["wmt"]="de-en en-de ru-en en-ru zh-en en-zh"
    ["law"]="de-en en-de"
    ["medical"]="de-en en-de"
    ["tico"]="en-fr en-pt"
    ["chat_wmt"]="en-de en-fr en-pt"
)

for DATASET in "${DATASETS[@]}"; do
    DATASET_DIR=$DATA_DIR/$DATASET
    LPS="${DATASETS_LPS[$DATASET]}"
    for LP in $LPS; do

        echo "Translating for LP $DATASET/$LP" 

        ZERO_SHOT_OUTPUT_DIR=$OUTPUT_DIR/$DATASET/$LP/$CKPT_NUM/zero_shot_instructions/
        FEW_SHOT_OUTPUT_DIR=$OUTPUT_DIR/$DATASET/$LP/$CKPT_NUM/few_shot_instructions2/

        mkdir -p $ZERO_SHOT_OUTPUT_DIR
        python translation_llm/generate_lora.py \
            --prompts_path $DATASET_DIR/$LP/zero_shot_instructions.txt \
            --output_path $ZERO_SHOT_OUTPUT_DIR/translations.txt \
            --tokenizer_path $TOKENIZER_PATH \
            --tokenizer_path $TOKENIZER_PATH \
            --pretrained_path $CONVERTED_LLAMA_MODEL \
            --lora_cfg_path $CFG_PATH \
            --lora_path $CKPT_PATH \
            --max_new_tokens 200

        mkdir -p $FEW_SHOT_OUTPUT_DIR
        python translation_llm/generate_lora.py \
            --prompts_path $DATASET_DIR/$LP/few_shot_instructions2.txt \
            --output_path $FEW_SHOT_OUTPUT_DIR/translations.txt \
            --tokenizer_path $TOKENIZER_PATH \
            --pretrained_path $CONVERTED_LLAMA_MODEL \
            --lora_cfg_path $CFG_PATH \
            --lora_path $CKPT_PATH \
            --max_new_tokens 200

    done
done