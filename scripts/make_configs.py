"""Generate every YAML config for the experiment grid.

Core matrix: 4 models x 2 methods x 3 tasks x 3 seeds = 72 runs.
Rank ablation: xlmr-base + LoRA on coreference, r in {4,8,16,32},
seeds {42,43} (r=8 seed 42 overlaps the core grid and is reused).

Fixed-budget fairness (as promised in the proposal): epochs, batch
size, and max length are IDENTICAL across methods within a task; only
the learning rate differs by method, set to each method's published
standard (2e-5 full fine-tuning, 2e-4 LoRA per Hu et al.). Record
this choice verbatim in the report's protocol section.
"""

from itertools import product
from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parent.parent / "configs" / "grid"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["nepberta", "iriis-roberta", "mbert", "xlmr-base"]
METHODS = ["full", "lora"]
SEEDS = [42, 43, 44]

# Per-task training budget (identical across methods and models).
TASK_BUDGET = {
    #                epochs  batch  grad_accum
    "sentiment":     dict(epochs=3,  batch_size=32, grad_accum=1),
    "acceptability": dict(epochs=5,  batch_size=32, grad_accum=1),
    "coreference":   dict(epochs=15, batch_size=16, grad_accum=1),
}

LR = {"full": 2e-5, "lora": 2e-4}


def write(cfg: dict, name: str) -> None:
    (OUT / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def core_grid() -> int:
    n = 0
    for model, method, task, seed in product(
            MODELS, METHODS, TASK_BUDGET, SEEDS):
        cfg = dict(model=model, task=task, method=method, seed=seed,
                   lr=LR[method], **TASK_BUDGET[task])
        if method == "lora":
            cfg.update(lora_r=8, lora_alpha=16)
        write(cfg, f"{model}_{task}_{method}_s{seed}")
        n += 1
    return n


def rank_ablation() -> int:
    n = 0
    for r, seed in product([4, 8, 16, 32], [42, 43]):
        if r == 8 and seed == 42:
            continue  # already in the core grid
        cfg = dict(model="xlmr-base", task="coreference", method="lora",
                   seed=seed, lr=LR["lora"], lora_r=r, lora_alpha=2 * r,
                   **TASK_BUDGET["coreference"])
        write(cfg, f"ablation_xlmr_coref_r{r}_s{seed}")
        n += 1
    return n


if __name__ == "__main__":
    total = core_grid() + rank_ablation()
    print(f"Wrote {total} configs to {OUT}")
