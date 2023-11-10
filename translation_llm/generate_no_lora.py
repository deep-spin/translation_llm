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
    ckpt_path: Path,
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
        ckpt_path: The path to the checkpoint with the LLaMA weights.
        tokenizer_path: The tokenizer path to load.
        dtype: The dtype to use during generation.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
    """
    assert prompts_path.is_file()
    assert ckpt_path.is_file()
    assert tokenizer_path.is_file()

    fabric = L.Fabric(devices=1)

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    prompts = files.read_lines(prompts_path, unescape_newline=True)
    if head is not None:
        prompts = prompts[:head]


    with utils.lazy_load(ckpt_path) as ckpt:
        name = utils.llama_model_lookup(ckpt)

        with utils.EmptyInitOnDevice(device=fabric.device, dtype=dtype):
            model = LLaMA.from_name(name)

            # 1. Load the pretrained weights
            model.load_state_dict(ckpt, strict=True)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)
    outputs = []
    
    for prompt in tqdm(prompts):
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

        output = generate.greedy(
            model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eos_id
        )

        output = tokenizer.decode(output)
        outputs.append(output)

    files.write_lines(output_path, outputs, escape_newline=True)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI([main], as_positional=False)