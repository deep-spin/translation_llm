"""
Instruction-tuning on the Alpaca dataset using a regular finetuning procedure (updating all layers).

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
from lightning.fabric.strategies import FSDPStrategy
from translation_llm import data
from translation_llm import train_eval
from translation_llm.model import LLaMA, LLaMAConfig, Block
from translation_llm.tokenizer import Tokenizer
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
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
    
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing=Block)

    fabric = L.Fabric(accelerator="cuda", devices="auto", precision="bf16-mixed", strategy=strategy)
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
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
        max_seq_length=max_seq_length,
    )

    train_step_func = partial(
        train_eval.train_step, fabric, model, optimizer, scheduler, tokenize_func, label_smoothing,
    )
    validate_func = partial(train_eval.validate, fabric, model, tokenize_func, val_samples, val_batch_size,)
    
    wandb_config = {
        "lora": False,
        "model_size": model_size,
        "train_batch_size": train_batch_size,
        "grad_accumulation_steps": grad_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "label_smoothing": label_smoothing,
    }
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(wandb_config, f, indent=4)

    disable_wandb = disable_wandb or fabric.global_rank != 0
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
    max_seq_length: int,
) -> Tuple[LLaMA, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    config = LLaMAConfig.from_name(model_size)
    config.block_size = max_seq_length

    checkpoint = torch.load(pretrained_path)

    with fabric.device:
        torch.set_default_tensor_type(torch.HalfTensor)
        model = LLaMA(config).bfloat16()
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint) 

    model = fabric.setup_module(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    optimizer = fabric.setup_optimizers(optimizer)

    lr_schedule_func = partial(
        train_eval.constant_schedule, warmup_steps=warmup_steps, total_steps=total_steps,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_func)

    return model, optimizer, scheduler

def save_model(
    fabric: L.Fabric,
    model: LLaMA,
    ckpt_dir: Path,
    save_at: List[int],
    step: int,):
    """Handles boilerplate logic for retrieving and saving the state_dict.
    
    This will be upstreamed to Fabric soon.
    """

    if (step + 1) in save_at:
        ckpt_path = ckpt_dir / f"model-{step+1}.pt"
        save_policy = FullStateDictConfig(offload_to_cpu=(fabric.world_size > 1), rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model._forward_module.state_dict()

        if fabric.global_rank == 0:
            torch.save(state_dict, ckpt_path)
        fabric.barrier()


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse.cli import CLI

    CLI([main], as_positional=False)