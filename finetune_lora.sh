#!/bin/bash

CONFIG="./config.sh"

[[ ! -f "${CONFIG}" ]] && echo "${CONFIG} not found: Create ${CONFIG} from config.example.sh with your configurations." && exit 1

# Import configurations and export for script.
set -a
source ${CONFIG}
set +a

[[ -z "$CONVERTED_DIR" ]] && echo "CONVERTED_DIR not set in config.sh" && exit 1
[[ -z "$VENV" ]]     && echo "VENV not set in config.sh" && exit 1

DATA_DIR=<path to data dir>
TRAIN_DATA_DIR=$DATA_DIR/few_shot_balanced_uniform_train_data
VALID_DATA_DIR=$DATA_DIR/few_shot_balanced_uniform_val_data

MODEL_SIZE=<model size>
CONVERTED_LLAMA_MODEL=$CONVERTED_DIR/$MODEL_SIZE/state_dict.pth
TOKENIZER_PATH=$CONVERTED_DIR/tokenizer.model
SAVE_DIR=<path to save run results>

source $VENV/bin/activate

python translation_llm/finetune_lora.py \
        --train_prompts_path $TRAIN_DATA_DIR/few_shot_instructions_2.txt \
        --train_targets_path $TRAIN_DATA_DIR/references.txt \
        --val_prompts_path $VALID_DATA_DIR/few_shot_instructions_2.txt \
        --val_targets_path $VALID_DATA_DIR/references.txt \
        --model_size $MODEL_SIZE \
        --tokenizer_path $TOKENIZER_PATH \
        --pretrained_path $CONVERTED_LLAMA_MODEL \
        --ckpt_dir $SAVE_DIR \
        --warmup_steps 500 \
        --total_steps 25000 \
        --train_batch_size 8 \
        --val_every 400 \
        --val_batch_size 64 \
        --learning_rate 2e-04 \
        --weight_decay 0.0 \
        --label_smoothing 0.001 \
        --grad_accumulation_steps 1 \
        --lora_r 256 \
        --lora_alpha 512 \
        --lora_dropout 0.05 \
        --mask_prompt False \
        --save_at [1000,2000,5000,10000,15000,20000,25000]

