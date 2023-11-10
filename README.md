# Steering Large Language Models for Machine Translation with Finetuning and In-Context Learning

This repository provides the code for the paper "Steering Large Language Models for Machine Translation with Finetuning and In-Context Learning".
It was based on the [lit-llama](https://github.com/Lightning-AI/lit-llama) repository.

## Quick Installation
To install run:
```bash
pip install -r requirements.txt
pip install -e .
```

## Convert LLaMA checkpoint

Before fine-tuning the pre-trained model, you need to convert the original LLaMA checkpoint to be compatible with the tool. To do this, run:

```bash
python translation_llm/convert_checkpoint.py \
    --output_dir <directory to save converted checkpoint> \
    --checkpoint_dir <directory with original checkpoint> \
    --model_size <7B/13B>
```

## Finetune

Look at the `finetune_lora.sh` and `finetune_no_lora.sh` scripts for examples of how to finetune the models.

## Generate

Look at the `generate_lora.sh` and `generate_no_lora.sh` scripts for examples of how to generate from the models.

## Evaluate

For the evaluation environment, you need to install the packages `sacrebleu` and `unbabel-comet`.

Look at the `eval.sh` script for an example on how to evaluate the models.
