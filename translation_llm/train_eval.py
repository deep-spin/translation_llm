import lightning as L
import torch.optim
import torch

from torch import Tensor
from translation_llm import data
from translation_llm.model import LLaMA

from typing import List, Tuple

def train_step(
    fabric: L.Fabric,
    model: LLaMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    tokenize_func: data.Tokenizer,
    label_smoothing: float,
    samples: List[data.Sample],
    is_grad_accumulation_step: bool,
    mask_prompt: bool,
) -> float:
    prompts_ids, targets_ids = prepare_batch(
        fabric, tokenize_func, samples, mask_prompt=mask_prompt,
    )
    logits = model(prompts_ids)
    loss = loss_fn(logits, targets_ids, label_smoothing)
    fabric.backward(loss)

    if not is_grad_accumulation_step:
        optimizer.step()
        optimizer.zero_grad()

    # Always step as the parameters are passed based on the total number of steps.
    #
    # This will lead to a warning at the beginning of training if grad 
    # accumulation is enabled as we don't perform the first step.
    #
    # In order to disable this warning, we need to move the scheduler step
    # into the if statement above, but this will require changing the
    # warmup_steps and total_steps to be based on the number of grad accumulation steps.
    scheduler.step()
    return loss.item()

@torch.no_grad()
def validate(
    fabric: L.Fabric, 
    model: torch.nn.Module, 
    tokenize_func: data.Tokenizer,
    val_data: List[data.Sample],
    batch_size: int,
    mask_prompt: bool,
) -> float:

    model.eval()
    losses = []
    for i in range(0, len(val_data), batch_size):
        samples = val_data[i : i + batch_size]
        prompts_ids, targets_ids = prepare_batch(fabric, tokenize_func, samples, mask_prompt=mask_prompt)
        logits = model(prompts_ids)
        loss = loss_fn(logits, targets_ids, label_smoothing=0.0)
        losses.append(loss.item())

    model.train()
    return torch.mean(torch.tensor(losses)).item()

def prepare_batch(
    fabric: L.Fabric,
    tokenize_func: data.Tokenizer,
    samples: List[data.Sample],
    mask_prompt: bool,
) -> Tuple[Tensor, Tensor]:
    tokenized = data.tokenize_samples(tokenize_func, samples, mask_prompt=mask_prompt)
    batch = data.batch_samples(tokenized)
    prompts_ids, targets_ids, _, _ = batch
    
    pinned = (prompts_ids.pin_memory(), targets_ids.pin_memory())
    prompts_ids, targets_ids = fabric.to_device(pinned)

    return prompts_ids, targets_ids

def loss_fn(logits, targets, label_smoothing: float):
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), 
        targets.view(-1),
        ignore_index=-1,
        label_smoothing=label_smoothing
    )
    return loss

def linear_schedule(step: int, total_steps: int, warmup_steps: int = 0) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    step_no_warmup = step - warmup_steps
    total_no_warmup = total_steps - warmup_steps
    progress = step_no_warmup / total_no_warmup
    return max(0, 1 - progress)

def constant_schedule(step: int, total_steps: int, warmup_steps: int = 0) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0