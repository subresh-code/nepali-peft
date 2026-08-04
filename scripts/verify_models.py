"""Run this ONCE before the grid (5 min on Colab).

Settles empirically, on your runtime, the two open items from the
proposal checklist: (1) the exact checkpoint IDs resolve, and (2) the
true parameter count of the IRIIS RoBERTa (repo named 125M, card text
says 110M). Whatever this prints is ground truth for your report.
Also prints tokenizer vocab size — the seed of your fertility
analysis later.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.modeling import MODELS

if __name__ == "__main__":
    print(f"{'key':<14}{'checkpoint':<42}{'params':>12}{'vocab':>9}")
    print("-" * 77)
    for key, (ckpt, family) in MODELS.items():
        try:
            tok = AutoTokenizer.from_pretrained(ckpt)
            model = AutoModelForSequenceClassification.from_pretrained(
                ckpt, num_labels=2)
            n = sum(p.numel() for p in model.parameters())
            print(f"{key:<14}{ckpt:<42}{n/1e6:>10.1f}M{len(tok):>9}")
            del model
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"{key:<14}{ckpt:<42}  FAILED: {exc}")
    print("\nIf a Nepali checkpoint FAILED, search its org page on the "
          "HF Hub for the correct repo name and update src/modeling.py.")
