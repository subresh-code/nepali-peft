"""One experimental run = one YAML config = one row in results.csv.

Usage:
    python train.py --config configs/smoke_test.yaml

Every run records accuracy metrics (val + test) AND the efficiency
axis (trainable params, peak VRAM, wall-clock) that the proposal's
H1/H3 and the Pareto figure depend on. Checkpointing is enabled so a
Colab preemption costs at most one epoch.
"""

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from transformers import (DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)

from src.data import load_task, tokenize
from src.metrics import build_compute_metrics
from src.modeling import build, family_of

RESULTS_CSV = Path("results/results.csv")

CSV_FIELDS = [
    "timestamp", "run_id", "model", "family", "task", "method",
    "lora_r", "seed", "epochs", "lr",
    "val_macro_f1", "val_accuracy", "val_mcc",
    "test_macro_f1", "test_accuracy", "test_mcc",
    "trainable_params", "total_params", "pct_trainable",
    "peak_vram_gb", "wall_clock_min",
]


def append_result(row: dict) -> None:
    RESULTS_CSV.parent.mkdir(exist_ok=True)
    new_file = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    set_seed(cfg["seed"])

    ds, spec = load_task(cfg["task"])
    built = build(
        cfg["model"], cfg["method"], spec.num_labels,
        lora_r=cfg.get("lora_r", 8),
        lora_alpha=cfg.get("lora_alpha", 16),
    )
    ds = tokenize(ds, spec, built.tokenizer)

    run_id = (f"{cfg['model']}_{cfg['task']}_{cfg['method']}"
              f"_r{cfg.get('lora_r', 0) if cfg['method'] == 'lora' else 0}"
              f"_s{cfg['seed']}")

    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=f"checkpoints/{run_id}",
        seed=cfg["seed"],
        num_train_epochs=cfg["epochs"],
        learning_rate=float(cfg["lr"]),
        per_device_train_batch_size=cfg.get("batch_size", 16),
        per_device_eval_batch_size=cfg.get("eval_batch_size", 64),
        gradient_accumulation_steps=cfg.get("grad_accum", 1),
        fp16=use_cuda,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=built.model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=DataCollatorWithPadding(built.tokenizer),
        compute_metrics=build_compute_metrics(spec.report_mcc),
    )

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    wall_min = (time.time() - t0) / 60
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if use_cuda else 0.0

    val = trainer.evaluate(ds["validation"], metric_key_prefix="val")
    test = trainer.evaluate(ds["test"], metric_key_prefix="test")

    if cfg["method"] == "lora":
        built.model.save_pretrained(f"adapters/{run_id}")

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "model": cfg["model"],
        "family": family_of(cfg["model"]),
        "task": cfg["task"],
        "method": cfg["method"],
        "lora_r": cfg.get("lora_r", "") if cfg["method"] == "lora" else "",
        "seed": cfg["seed"],
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "val_macro_f1": round(val["val_macro_f1"], 4),
        "val_accuracy": round(val["val_accuracy"], 4),
        "val_mcc": round(val.get("val_mcc", float("nan")), 4)
                   if "val_mcc" in val else "",
        "test_macro_f1": round(test["test_macro_f1"], 4),
        "test_accuracy": round(test["test_accuracy"], 4),
        "test_mcc": round(test.get("test_mcc", float("nan")), 4)
                    if "test_mcc" in test else "",
        "trainable_params": built.trainable_params,
        "total_params": built.total_params,
        "pct_trainable": round(
            100 * built.trainable_params / built.total_params, 3),
        "peak_vram_gb": round(peak_gb, 2),
        "wall_clock_min": round(wall_min, 1),
    }
    append_result(row)
    print("\n=== RUN COMPLETE ===")
    for k in ("run_id", "test_macro_f1", "test_accuracy",
              "pct_trainable", "peak_vram_gb", "wall_clock_min"):
        print(f"  {k}: {row[k]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args().config)
