"""Model construction for the 2x2 design (family x adaptation method).

Checkpoint IDs: mBERT and XLM-R are canonical. The two Nepali IDs are
the best-known candidates as of Aug 2026 — run scripts/verify_models.py
ONCE before the grid to confirm they resolve and to record true
parameter counts (the IRIIS repo is named 125M while its card text
says 110M; the verify script settles it empirically, and whatever it
prints is what goes in your report).
"""

from dataclasses import dataclass

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS = {
    # id -> (hf_checkpoint, family)
    "nepberta": ("NepBERTa/NepBERTa", "monolingual"),
    "iriis-roberta": ("IRIIS-RESEARCH/RoBERTa_Nepali_125M", "monolingual"),
    "mbert": ("bert-base-multilingual-cased", "multilingual"),
    "xlmr-base": ("xlm-roberta-base", "multilingual"),
}

# BERT- and RoBERTa-family attention projections share these names in
# HF implementations; LoRA targets query and value per Hu et al. (2022).
LORA_TARGET_MODULES = ["query", "value"]


@dataclass
class BuiltModel:
    model: object
    tokenizer: object
    trainable_params: int
    total_params: int


def _count_params(model) -> tuple:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def build(model_key: str, method: str, num_labels: int,
          lora_r: int = 8, lora_alpha: int = 16,
          lora_dropout: float = 0.05) -> BuiltModel:
    """method: 'full' or 'lora'."""
    checkpoint, _family = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels
    )

    if method == "lora":
        cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,   # keeps the classifier trainable
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=LORA_TARGET_MODULES,
        )
        model = get_peft_model(model, cfg)
    elif method != "full":
        raise ValueError(f"Unknown adaptation method: {method}")

    trainable, total = _count_params(model)
    return BuiltModel(model, tokenizer, trainable, total)


def family_of(model_key: str) -> str:
    return MODELS[model_key][1]
