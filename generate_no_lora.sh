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
# Change path for other checkpoints
CONVERTED_LLAMA_MODEL=$CONVERTED_DIR/$MODEL_SIZE/state_dict.pth
TOKENIZER_PATH=$CONVERTED_DIR/tokenizer.model

source $VENV/bin/activate

DATA_DIR=<path to data>
OUTPUT_DIR=<path to store translations>

for DATASET in "flores"; do
    DATASET_DIR=$DATA_DIR/$DATASET
    for LP in "de-en" "en-de" "fr-en" "en-fr" "nl-en" "en-nl" "pt-en" "en-pt" "ru-en" "en-ru" "zh-en" "en-zh"; do

        echo "Translating for LP $DATASET/$LP"

        CKPT_NUM=0

        ZERO_SHOT_OUTPUT_DIR=$OUTPUT_DIR/$DATASET/$LP/$CKPT_NUM/zero_shot_instructions/
        FEW_SHOT_OUTPUT_DIR=$OUTPUT_DIR/$DATASET/$LP/$CKPT_NUM/few_shot_instructions2/

        mkdir -p $ZERO_SHOT_OUTPUT_DIR
        python translation_llm/generate_no_lora.py \
            --prompts_path $DATASET_DIR/$LP/zero_shot_instructions.txt \
            --output_path $ZERO_SHOT_OUTPUT_DIR/translations.txt \
            --tokenizer_path $TOKENIZER_PATH \
            --ckpt_path $CONVERTED_LLAMA_MODEL \
            --max_new_tokens 200

        mkdir -p $FEW_SHOT_OUTPUT_DIR
        python translation_llm/generate_no_lora.py \
            --prompts_path $DATASET_DIR/$LP/few_shot_instructions2.txt \
            --output_path $FEW_SHOT_OUTPUT_DIR/translations.txt \
            --tokenizer_path $TOKENIZER_PATH \
            --ckpt_path $CONVERTED_LLAMA_MODEL \
            --max_new_tokens 200

    done
done
