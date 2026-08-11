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
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import src.telemetry  # noqa: F401  must precede torch/transformers: wires OTel

import torch
import yaml
from opentelemetry import metrics, trace
from transformers import (DataCollatorWithPadding, Trainer,
                          TrainingArguments, set_seed)

from src.data import load_task, tokenize
from src.metrics import build_compute_metrics
from src.modeling import build, family_of

log = logging.getLogger("nepali_peft.train")
tracer = trace.get_tracer("nepali_peft.train")
meter = metrics.get_meter("nepali_peft.train")

runs_completed = meter.create_counter("train.runs.completed", unit="1")
wall_clock_hist = meter.create_histogram("train.wall_clock_minutes", unit="min")
test_f1_hist = meter.create_histogram("train.test_macro_f1", unit="1")

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


@tracer.start_as_current_span("train.run")
def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())

    run_id = (f"{cfg['model']}_{cfg['task']}_{cfg['method']}"
              f"_r{cfg.get('lora_r', 0) if cfg['method'] == 'lora' else 0}"
              f"_s{cfg['seed']}")
    span = trace.get_current_span()
    span.set_attributes({
        "run.id": run_id, "model.key": cfg["model"], "task": cfg["task"],
        "method": cfg["method"], "seed": cfg["seed"],
        "epochs": cfg["epochs"], "lr": str(cfg["lr"]),
        **({"lora.r": cfg.get("lora_r", 8)} if cfg["method"] == "lora" else {}),
    })

    # ponytail: substring check, fine while run_ids stay unambiguous
    if RESULTS_CSV.exists() and run_id in RESULTS_CSV.read_text():
        span.set_attribute("run.skipped", True)
        log.info("skip %s: already in %s", run_id, RESULTS_CSV)
        print(f"SKIP {run_id}: already in {RESULTS_CSV}")
        return

    log.info("starting run %s", run_id)
    set_seed(cfg["seed"])

    ds, spec = load_task(cfg["task"])
    built = build(
        cfg["model"], cfg["method"], spec.num_labels,
        lora_r=cfg.get("lora_r", 8),
        lora_alpha=cfg.get("lora_alpha", 16),
    )
    ds = tokenize(ds, spec, built.tokenizer)

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
    with tracer.start_as_current_span("train.fit"):
        trainer.train()
    wall_min = (time.time() - t0) / 60
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if use_cuda else 0.0

    with tracer.start_as_current_span("train.evaluate"):
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

    span.set_attributes({
        "result.val_macro_f1": row["val_macro_f1"],
        "result.test_macro_f1": row["test_macro_f1"],
        "result.test_accuracy": row["test_accuracy"],
        "result.trainable_params": row["trainable_params"],
        "result.peak_vram_gb": row["peak_vram_gb"],
        "result.wall_clock_min": row["wall_clock_min"],
    })
    dims = {"model": cfg["model"], "task": cfg["task"], "method": cfg["method"]}
    runs_completed.add(1, dims)
    wall_clock_hist.record(row["wall_clock_min"], dims)
    test_f1_hist.record(row["test_macro_f1"], dims)
    log.info("run %s complete: test_macro_f1=%s wall_clock_min=%s",
             run_id, row["test_macro_f1"], row["wall_clock_min"])

    print("\n=== RUN COMPLETE ===")
    for k in ("run_id", "test_macro_f1", "test_accuracy",
              "pct_trainable", "peak_vram_gb", "wall_clock_min"):
        print(f"  {k}: {row[k]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    main(p.parse_args().config)
