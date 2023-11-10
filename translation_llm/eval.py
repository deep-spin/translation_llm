import comet
import pandas as pd
import os
import torch

from jsonargparse import CLI
from pathlib import Path
from sacrebleu import BLEU, CHRF

def main(
    sources_path: str,
    translations_path: str,
    references_path: str,
    sys_scores_path: str,
    seg_scores_path: str,
):
    
    sources_path = Path(sources_path)
    translations_path = Path(translations_path)
    references_path = Path(references_path)
    sys_scores_path = Path(sys_scores_path)
    seg_scores_path = Path(seg_scores_path)

    sources = read_lines(sources_path)
    references = read_lines(references_path)
    translations = read_lines(translations_path, unescape_newline=True)

    # Clean until first newline
    translations = [t.strip().split("\n")[0] for t in translations]

    tokenize = "zh" if "en-zh" in str(translations_path) else None
    bleu_sys_score, bleu_seg_scores = run_bleu(translations, references, tokenize=tokenize)
    chrf_sys_score, chrf_seg_scores = run_chrf(translations, references)

    comet_kiwi_sys_score, comet_kiwi_seg_scores = run_comet_kiwi(sources, translations, references)
    comet_sys_score, comet_seg_scores = run_comet(sources, translations, references)

    print(
        "COMET-22:", comet_sys_score,
        "COMETKiwi:", comet_kiwi_sys_score,
        "BLEU:", bleu_sys_score,
        "CHRF:", chrf_sys_score,
    )

    lines = [
        f"COMET-22: {comet_sys_score}",
        f"COMETKiwi: {comet_kiwi_sys_score}",
        f"BLEU: {bleu_sys_score}",
        f"chrF: {chrf_sys_score}",
    ]

    seg_scores = pd.DataFrame({
        "COMET-22": comet_seg_scores,
        "COMETKiwi": comet_kiwi_seg_scores,
        "BLEU": bleu_seg_scores,
        "chrF": chrf_seg_scores,
    })

    print(seg_scores.describe())

    with open(sys_scores_path, "w") as f:
        f.writelines((f"{l}\n" for l in lines))

    seg_scores.to_csv(seg_scores_path, index=False)


def run_bleu(translations, references, tokenize=None):
    bleu = BLEU(tokenize=tokenize)
    sys_score = bleu.corpus_score(translations, [references]).score
    bleu = BLEU(tokenize=tokenize, effective_order=True)
    seg_scores = [bleu.sentence_score(t, [r]).score for t, r in zip(translations, references)]
    return sys_score, seg_scores


def run_chrf(translations, references):
    chrf = CHRF()
    sys_score = chrf.corpus_score(translations, [references]).score
    seg_scores = [chrf.sentence_score(t, [r]).score for t, r in zip(translations, references)]
    return sys_score, seg_scores


def run_comet(sources, translations, references):
    comet_path = comet.download_model("Unbabel/wmt22-comet-da")
    comet_model = comet.load_from_checkpoint(comet_path)

    comet_input = [
        {"src": s, "mt": m, "ref": r}
        for s, m, r in zip(sources, translations, references)
    ]
    batch_size = 8 if torch.cuda.is_available() else 1
    gpus = 1 if torch.cuda.is_available() else 0
    model_output = comet_model.predict(comet_input, batch_size=batch_size, gpus=gpus)
    return model_output["system_score"], model_output["scores"]

def run_comet_kiwi(sources, translations, references):
    from huggingface_hub import login

    login_token = os.environ["HF_LOGIN"]
    login(token=login_token)
    comet_path = comet.download_model("Unbabel/wmt22-cometkiwi-da")
    comet_model = comet.load_from_checkpoint(comet_path)

    comet_input = [
        {"src": s, "mt": m, "ref": r}
        for s, m, r in zip(sources, translations, references)
    ]
    batch_size = 8 if torch.cuda.is_available() else 1
    gpus = 1 if torch.cuda.is_available() else 0
    model_output = comet_model.predict(comet_input, batch_size=batch_size, gpus=gpus)
    return model_output["system_score"], model_output["scores"]

def read_lines(path: Path, unescape_newline: bool = False):
    """Reads lines from a file, removing the newline character at the end of each line.

    Does not use the function from translation_llm.files because it
    adds the translation_llm as dependency to this script.
    """
    with open(path) as f:
        lines = [l[:-1] for l in f.readlines()]
    if unescape_newline:
        lines = [l.replace("\\n", "\n") for l in lines]
    return lines

if __name__ == "__main__":
    CLI([main], as_positional=False)
