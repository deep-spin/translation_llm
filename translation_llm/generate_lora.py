import json
import lightning as L
import torch
import warnings

from jsonargparse import CLI
from pathlib import Path
from translation_llm import files
from translation_llm import generate
from translation_llm import lora
from translation_llm import utils
from translation_llm.model import LLaMA
from translation_llm.tokenizer import Tokenizer
from tqdm import tqdm

from typing import Optional

def main(
    prompts_path: Path,
    output_path: Path,
    lora_path: Path,
    lora_cfg_path: Path,
    pretrained_path: Path,
    tokenizer_path: Path,
    dtype: str = "bfloat16",
    max_new_tokens: int = 100,
    head: Optional[int] = None,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned LoRA model.
    See `finetune_lora.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        lora_path: Path to the checkpoint with trained LoRA weights, which are the output of
            `finetune_lora.py`.
        input: Optional input (Alpaca style).
        pretrained_path: The path to the checkpoint with pretrained LLaMA weights.
        tokenizer_path: The tokenizer path to load.
        dtype: The dtype to use during generation.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
    """
    assert prompts_path.is_file()
    assert lora_path.is_file()
    assert lora_cfg_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    fabric = L.Fabric(devices=1)

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    prompts = files.read_lines(prompts_path, unescape_newline=True)
    if head is not None:
        prompts = prompts[:head]

    with open(lora_cfg_path, "r") as f:
        model_cfg = json.load(f)
    lora_r = model_cfg["lora_r"]
    lora_alpha = model_cfg["lora_alpha"]
    lora_dropout = model_cfg["lora_dropout"]

    with utils.lazy_load(pretrained_path) as pretrained_checkpoint, utils.lazy_load(lora_path) as lora_checkpoint:
        name = utils.llama_model_lookup(pretrained_checkpoint)

        with utils.EmptyInitOnDevice(
            device=fabric.device, dtype=dtype,
        ), lora.apply(r=lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
            model = LLaMA.from_name(name)

            # 1. Load the pretrained weights
            model.load_state_dict(pretrained_checkpoint, strict=False)
            # 2. Load the fine-tuned lora weights
            model.load_state_dict(lora_checkpoint, strict=False)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)
    outputs = []
    
    for i, prompt in enumerate(tqdm(prompts)):
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

        output = generate.greedy(
            model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eos_id
        )

        output = tokenizer.decode(output)
        outputs.append(output)

        if (i + 1) % 20 == 0:
            files.write_lines(output_path, outputs, escape_newline=True)

    files.write_lines(output_path, outputs, escape_newline=True)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI([main], as_positional=False)