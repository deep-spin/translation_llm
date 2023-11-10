#!/bin/bash

CONFIG="./config.sh"

[[ ! -f "${CONFIG}" ]] && echo "${CONFIG} not found: Create ${CONFIG} from config.example.sh with your configurations." && exit 1

# Import configurations and export for script.
set -a
source ${CONFIG}
set +a

[[ -z "$EVAL_VENV" ]]     && echo "EVAL_VENV not set in config.sh" && exit 1
[[ -z "$HF_LOGIN" ]]      && echo "HF_LOGIN not set in config.sh" && exit 1

source $EVAL_VENV/bin/activate

DATA_DIR=<path to data>
RESULTS_DIR=<path to all translations>

MODELS=<list of models>
declare -A MODEL_CKPS=(
    <model name>=<list of checkpoints>
)
DATASETS=(flores wmt law medical tico chat_wmt)
declare -A DATASETS_LPS=(
    ["flores"]="de-en en-de fr-en en-fr nl-en en-nl pt-en en-pt ru-en en-ru zh-en en-zh"
    ["wmt"]="de-en en-de ru-en en-ru zh-en en-zh"
    ["law"]="de-en en-de"
    ["medical"]="de-en en-de"
    ["tico"]="en-fr en-pt"
    ["chat_wmt"]="en-de en-fr en-pt"
)

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        LPS="${DATASETS_LPS[$DATASET]}"
        for LP in $LPS; do
            CKPTS="${MODEL_CKPS[$MODEL]}"
            for CKPT in $CKPTS; do
                echo "Evaluating $MODEL/$DATASET/$LP/$CKPT"
                
                SOURCES_PATH=$DATA_DIR/$DATASET/$LP/train_eval.input.txt
                REFERENCES_PATH=$DATA_DIR/$DATASET/$LP/train_eval.output.txt

                CKPT_DIR=$RESULTS_DIR/$MODEL/$DATASET/$LP/$CKPT
                ZERO_SHOT_INSTRUCTIONS_DIR=$CKPT_DIR/zero_shot_instructions
                FEW_SHOT_INSTRUCTIONS_DIR=$CKPT_DIR/few_shot_instructions2
                
                python translation_llm/eval.py \
                    --sources_path $SOURCES_PATH \
                    --translations_path $ZERO_SHOT_INSTRUCTIONS_DIR/translations.txt \
                    --references_path $REFERENCES_PATH \
                    --sys_scores_path $ZERO_SHOT_INSTRUCTIONS_DIR/sys_scores.txt \
                    --seg_scores_path $ZERO_SHOT_INSTRUCTIONS_DIR/seg_scores.txt

                python translation_llm/eval.py \
                    --sources_path $SOURCES_PATH \
                    --translations_path $FEW_SHOT_INSTRUCTIONS_DIR/translations.txt \
                    --references_path $REFERENCES_PATH \
                    --sys_scores_path $FEW_SHOT_INSTRUCTIONS_DIR/sys_scores.txt \
                    --seg_scores_path $FEW_SHOT_INSTRUCTIONS_DIR/seg_scores.txt
            done
        done
    done
done