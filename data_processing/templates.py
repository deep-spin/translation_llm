from typing import List, Optional, Tuple

code2lang = {
    "de": "German",
    "fr": "French",
    "en": "English",
    "nl": "Dutch",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
}

def instruction_template(lp: str, source: str, target: Optional[str] = None) -> str:
    src_code, tgt_code = lp.split("-")
    src_lang = code2lang[src_code]
    tgt_lang = code2lang[tgt_code]
    if target is None:
        return f"Translate the source text from {src_lang} to {tgt_lang}.\nSource: {source}\nTarget:"
    else:
        return f"Translate the source text from {src_lang} to {tgt_lang}.\nSource: {source}\nTarget: {target}"

def format1_few_shot_instruction_template(
    lp: str, source: str, examples: List[Tuple[str, str]],
) -> str:
    """Produces few-shot prompts with the following template:
    
    Translate the source text from {src_lang} to {tgt_lang}.
    Source: {example_1_source}
    Target: {example_1_target}
    ...
    Translate the source text from {src_lang} to {tgt_lang}.
    Source: {example_k_source}
    Target: {example_k_target}
    Translate the source text from {src_lang} to {tgt_lang}.
    Source: {source}
    Target:
    """
    if len(examples) == 0:
        return instruction_template(lp, source)

    examples_prompts = [instruction_template(lp, s, t) for s, t in examples]

    prompt = instruction_template(lp, source)

    return "\n".join(examples_prompts + [prompt])

def format2_few_shot_instruction_template(
    lp: str, source: str, examples: List[Tuple[str, str]],
) -> str:
    """Produces few-shot prompts with the following template:
    
    Consider the following {k} translations from {src_lang} to {tgt_lang}.
    Example 1:
    Source: {example_1_source}
    Target: {example_1_target}

    ...
    Example {k}:
    Source: {example_k_source}
    Target: {example_k_target}

    Translate the source text from {src_lang} to {tgt_lang}.
    Source: {source}
    Target:
    """
    if len(examples) == 0:
        return instruction_template(lp, source)
    
    src_code, tgt_code = lp.split("-")
    src_lang = code2lang[src_code]
    tgt_lang = code2lang[tgt_code]
    examples_prefix = f"Consider the following {len(examples)} translations from {src_lang} to {tgt_lang}.\n"
    
    examples_prompts = [
        f"Example {i+1}:\nSource: {s}\nTarget: {t}\n" for i, (s, t) in enumerate(examples)
    ]

    prompt = instruction_template(lp, source)
    
    return examples_prefix + "\n".join(examples_prompts + [prompt])

def format3_few_shot_instruction_template(
    lp: str, source: str, examples: List[Tuple[str, str]],
) -> str:
    """Produces few-shot prompts with the following template:
    
    Consider the following translations from {src_lang} to {tgt_lang}.
    Source: {example_1_source}
    Target: {example_1_target}
    ...
    Source: {example_k_source}
    Target: {example_k_target}

    Translate the source text from {src_lang} to {tgt_lang}.
    Source: {source}
    Target:
    """
    if len(examples) == 0:
        return instruction_template(lp, source)
    
    src_code, tgt_code = lp.split("-")
    src_lang = code2lang[src_code]
    tgt_lang = code2lang[tgt_code]
    examples_prefix = f"Consider the following translations from {src_lang} to {tgt_lang}.\n"
    
    examples_prompts = [f"Source: {s}\nTarget: {t}" for s, t in examples]

    prompt = instruction_template(lp, source)
    
    return examples_prefix + "\n".join(examples_prompts + [prompt])
