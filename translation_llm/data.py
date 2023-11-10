import torch

from pathlib import Path
from translation_llm import files
from torch import Tensor

from typing import List, NamedTuple, Protocol

class Sample(NamedTuple):
    prompt: str
    target: str

class TokenizedSample(NamedTuple):
    prompt_ids: Tensor
    target_ids: Tensor
    prompt: str
    target: str

class Batch(NamedTuple):
    """Batch containing B tokenized samples with padding."""

    # Inputs for the model containing rows of the form
    # [BOS, *tokenize(prompt + target), X_PAD_ID...]
    # Shape (B, T)
    prompts_ids: Tensor
    # Targets for the model containing rows of the form
    # [*tokenize(prompt + target), EOS, Y_PAD_ID...]
    # Shape (B, T)
    targets_ids: Tensor
    # The original prompts
    prompts: List[str]
    # The original targets
    targets: List[str]

PROMPT_PAD_ID = 0
TARGET_PAD_ID = -1

def load_samples(prompts_path: Path, targets_path: Path) -> List[Sample]:
    prompts = files.read_lines(prompts_path, unescape_newline=True)
    targets = files.read_lines(targets_path, unescape_newline=True)

    assert len(prompts) == len(targets), \
        f"Length mismatch: {len(prompts)} prompts, {len(targets)} targets"
    return [Sample(prompt, target) for prompt, target in zip(prompts, targets)]

def sample(data: List[Sample], n: int) -> List[Sample]:
    """Samples a batch of samples from the data.
    
    Args:
        data: The data split to sample from.
        n: The number of samples to sample.
        
    Returns:
        The sampled batch.
    """
    idx = torch.randint(len(data), (n,))
    return [data[i] for i in idx]

class Tokenizer(Protocol):

    def __call__(self, string: str, bos: bool, eos: bool) -> Tensor:
        ...

def tokenize_samples(
    tokenize_func: Tokenizer, samples: List[Sample], mask_prompt: bool,
) -> List[TokenizedSample]:
    """Tokenizes a list of samples.
    
    Args:
        tokenize_func: The function to use for tokenization.
        samples: The samples to tokenize.
        
    Returns:
        The tokenized samples.
    """
    return [tokenize_sample(tokenize_func, sample, mask_prompt) for sample in samples]

def tokenize_sample(
    tokenize_func: Tokenizer, sample: Sample, mask_prompt: bool,
) -> TokenizedSample:
    """Tokenizes a sample.
    
    Args:
        tokenize_func: The function to use for tokenization.
        sample: The sample to tokenize.
        
    Returns:
        The tokenized sample.
    """
    prompt_and_target = f"{sample.prompt} {sample.target}"
    full_ids = tokenize_func(prompt_and_target, True, True)
    if not mask_prompt:
        return TokenizedSample(full_ids, full_ids, sample.prompt, sample.target)

    target_ids = full_ids.clone()
    prompt_ids = tokenize_func(f"{sample.prompt} ", True, False)
    target_ids[:len(prompt_ids)] = -1
    return TokenizedSample(full_ids, target_ids, sample.prompt, sample.target)

def batch_samples(
    samples: List[TokenizedSample],
    prompts_pad_id: int = PROMPT_PAD_ID,
    targets_pad_id: int = TARGET_PAD_ID,
) -> Batch:
    """Creates batches from samples.

    The batches can be provided to the model as input and target.
    
    Args:
        samples: The samples to batch.
        prompts_pad_id: The id to use for padding the model inputs.
        targets_pad_id: The id to use for padding the desired outputs.
        
    Returns:
        The batch with the given samples.
    """
    B = len(samples)
    assert B > 0, "No samples to batch"

    prompts_ids = [s.prompt_ids.type(torch.int64) for s in samples]
    targets_ids = [s.target_ids.type(torch.int64) for s in samples]
    prompts = [s.prompt for s in samples]
    targets = [s.target for s in samples]

    T = max(t.size(0) for t in prompts_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = T - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=prompts_pad_id) for x in prompts_ids])
    y = torch.stack([pad_right(x, pad_id=targets_pad_id) for x in targets_ids])
    return Batch(x, y, prompts, targets)
