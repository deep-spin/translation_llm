"""
Instruction-tuning with LoRA on the Alpaca dataset.

Note: If you run into a CUDA error "Expected is_sm80 to be true, but got false", uncomment the line
`torch.backends.cuda.enable_flash_sdp(False)` in the script below (see https://github.com/Lightning-AI/lit-llama/issues/101).
"""
import json
import lightning as L
import os
import torch
import torch.backends.cuda
import wandb

from functools import partial
from pathlib import Path
from translation_llm import data
from translation_llm import lora
from translation_llm import train_eval
from translation_llm.model import LLaMA, LLaMAConfig
from translation_llm.tokenizer import Tokenizer
from tqdm import trange

from typing import List, Tuple

def main(
    train_prompts_path: Path,
    train_targets_path: Path,
    val_prompts_path: Path,
    val_targets_path: Path,
    model_size: str,
    tokenizer_path: Path,
    pretrained_path: Path,
    ckpt_dir: Path,
    warmup_steps: int,
    total_steps: int,
    train_batch_size: int,
    val_batch_size: int,
    val_every: int,
    grad_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    mask_prompt: bool,
    max_seq_length: int = 2048,
    save_at: List[int] = [],
    disable_wandb: bool = False,
    enable_flash: bool = True,
):  
    torch.set_float32_matmul_precision("high")
    if not enable_flash:
        torch.backends.cuda.enable_flash_sdp(False)

    if len(save_at) == 0:
        print("WARNING: No checkpoints will be saved")

    print("Mask prompt:", mask_prompt)
    
    fabric = L.Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    assert fabric.world_size == 1, "This script only supports single-GPU training"

    os.makedirs(ckpt_dir, exist_ok=True)

    train_samples = data.load_samples(train_prompts_path, train_targets_path)
    val_samples = data.load_samples(val_prompts_path, val_targets_path)

    tokenizer = Tokenizer(tokenizer_path)
    tokenize_func = partial(tokenizer.encode, max_length=max_seq_length)

    model, optimizer, scheduler = init_model_optim_scheduler(
        fabric=fabric,
        pretrained_path=pretrained_path,
        model_size=model_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        max_seq_length=max_seq_length,
    )

    train_step_func = partial(
        train_eval.train_step, 
        fabric, model, optimizer, scheduler, tokenize_func, label_smoothing, mask_prompt=mask_prompt,
    )
    validate_func = partial(
        train_eval.validate, 
        fabric, model, tokenize_func, val_samples, val_batch_size, mask_prompt=mask_prompt,
    )

    wandb_config = {
        "lora": True,
        "model_size": model_size,
        "train_batch_size": train_batch_size,
        "grad_accumulation_steps": grad_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "label_smoothing": label_smoothing,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
    }

    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(wandb_config, f, indent=4)
    disable_wandb = disable_wandb or not fabric.is_global_zero
    wandb.init(
        project="translation-llm",
        mode="disabled" if disable_wandb else None,
        config=wandb_config
    )

    wandb.log({"val_loss": validate_func()}, step=0)

    save_func = partial(save_model, fabric, model, ckpt_dir, save_at)

    for step in trange(total_steps):
        samples = data.sample(train_samples, train_batch_size)
        is_grad_accumulation_step = (step + 1) % grad_accumulation_steps != 0
        train_loss = train_step_func(samples, is_grad_accumulation_step)

        wandb_metrics = {
            "train_loss": train_loss,
            "learning_rate": scheduler.get_last_lr()[0],
        }

        if (step + 1) % val_every == 0:
            wandb_metrics["val_loss"] = validate_func()

        wandb.log(wandb_metrics, step=step + 1)

        save_func(step)

    wandb.finish()


def init_model_optim_scheduler(
    fabric: L.Fabric,
    pretrained_path: Path,
    model_size: str,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    total_steps: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    max_seq_length: int,
) -> Tuple[LLaMA, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    config = LLaMAConfig.from_name(model_size)
    config.block_size = max_seq_length

    checkpoint = torch.load(pretrained_path)

    with fabric.device, lora.apply(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
        torch.set_default_tensor_type(torch.HalfTensor)
        model = LLaMA(config).bfloat16()
        # strict=False because missing keys due to LoRA weights not contained in checkpoint state
        model.load_state_dict(checkpoint, strict=False)
        torch.set_default_tensor_type(torch.FloatTensor)
    
    lora.freeze_non_lora_weights(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    llama_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" not in n)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA parameters: {lora_params}")
    print(f"LLaMA parameters: {llama_params}")
    print(f"LLaMA/LoRA parameters: {llama_params / lora_params:.2f}")
    print(f"Percentage of trainable parameters: {num_params / total_params * 100:.2f}%")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    model, optimizer = fabric.setup(model, optimizer)

    lr_schedule_func = partial(
        train_eval.linear_schedule, warmup_steps=warmup_steps, total_steps=total_steps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_func)

    return model, optimizer, scheduler


def save_model(
    fabric: L.Fabric,
    model: LLaMA,
    ckpt_dir: Path,
    save_at: List[int],
    step: int,
):
    if (step + 1) in save_at:
        ckpt_path = ckpt_dir / f"model-{step+1}.pt"
        state_dict = lora.state_dict(model)            
        fabric.save(ckpt_path, state_dict)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    
    from jsonargparse.cli import CLI

    CLI([main], as_positional=False)