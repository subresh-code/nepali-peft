"""Tokenizer fertility: mean subword tokens per whitespace word on Nepali text.

Sample = acceptability test split (1,950 single sentences). Lower fertility =
the tokenizer represents Nepali more compactly (more effective context, less
compute per sentence).

Usage: python scripts/fertility.py   (writes results/fertility.csv)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from transformers import AutoTokenizer

from src.data import load_task
from src.modeling import MODELS

ds, _ = load_task("acceptability")
sentences = ds["test"]["sentence"]
n_words = sum(len(s.split()) for s in sentences)

rows = []
for key, (ckpt, family) in MODELS.items():
    tok = AutoTokenizer.from_pretrained(ckpt)
    enc = tok(list(sentences), add_special_tokens=False)["input_ids"]
    n_tokens = sum(len(ids) for ids in enc)
    unk = sum(ids.count(tok.unk_token_id) for ids in enc) if tok.unk_token_id is not None else 0
    rows.append({
        "model": key, "family": family, "vocab_size": tok.vocab_size,
        "fertility": round(n_tokens / n_words, 3),
        "unk_pct": round(100 * unk / n_tokens, 3),
    })

df = pd.DataFrame(rows)
df.to_csv("results/fertility.csv", index=False)
print(f"{len(sentences)} sentences, {n_words} words")
print(df.to_string(index=False))
