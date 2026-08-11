# Worklog — what we did and why

Chronological record of the setup-and-verification session (2026-08-04 → 2026-08-09),
before the GPU training campaign. Each step: what happened, why it was necessary,
and what it settled for the report.

---

## 1. GitHub authentication fix

**What:** `gh` had two accounts configured (personal `subresh-code`, office org
`NIFN-dev`); the stored token was invalid. Re-authenticated `subresh-code` via
device-code flow. `NIFN-dev` deliberately left unconfigured for now.

**Why:** Couldn't push the repo without working auth. This project lives under the
personal account, so only `subresh-code` was needed. `gh auth switch -u <user>`
handles the two-account situation later if the office account is required.

## 2. Environment setup

**What:** Created `.venv` with CPU-only PyTorch, transformers 5.14.1, peft,
datasets, scikit-learn. This machine has no GPU (31 GB RAM, CPU only).

**Why:** Local machine is for harness development and smoke testing; the real
training campaign runs on Colab GPUs. Keeping a working local env means every
pipeline bug gets caught for free before spending GPU time.

## 3. Checkpoint verification (`scripts/verify_models.py`)

**What:** Ran the verify script that loads each of the four checkpoints and prints
parameter/vocab counts. Three passed; **NepBERTa failed** — the official
`NepBERTa/NepBERTa` repo ships only TensorFlow weights (`tf_model.h5`), and
transformers v5 removed TF-weight loading entirely (`from_tf=True` is gone).

**Why:** The proposal's 2×2 design (model family × adaptation method) collapses if
any checkpoint doesn't resolve. Verifying once, up front, also settles empirical
facts for the report — notably that IRIIS RoBERTa is **124.6M params** (the repo
name "125M" is right; the model card's "110M" is wrong).

Final verified table:

| key | checkpoint | params | vocab |
|---|---|---|---|
| nepberta | subrace/NepBERTa-pytorch | 109.5M | 30,523 |
| iriis-roberta | IRIIS-RESEARCH/RoBERTa_Nepali_125M | 124.6M | 50,256 |
| mbert | bert-base-multilingual-cased | 177.9M | 119,547 |
| xlmr-base | xlm-roberta-base | 278.0M | 250,002 |

## 4. The NepBERTa provenance problem → our own conversion

**What:** A community PyTorch port exists (`Rajan/nepbertaTorch`). We first
verified it (identical vocab md5, identical config) and pointed the harness at it —
but it's a third-party conversion with no documented process. Decision: **convert
the official TF weights ourselves** and publish under the user's HF account.

Conversion (`scripts/convert_nepberta.py`, committed):
- Separate venv with transformers 4.57.6 + tensorflow-cpu 2.21.0 (v5 can't read TF).
- `from_tf=True` in 4.57.6 was itself broken (loads onto meta-device skeleton, so
  weight copies silently no-op → "Cannot copy out of meta tensor" on save).
  Workaround: build `BertModel(cfg)` with real tensors, then call
  `load_tf2_checkpoint_in_pytorch_model(model, h5, allow_missing_keys=True)` directly.
- Verified tensor-by-tensor with `torch.equal` against the Rajan port:
  **bit-identical on every tensor the official checkpoint contains.**

**Key discovery:** the one mismatching tensor (`pooler.dense.weight`) led to
inspecting the h5 — the official checkpoint is an MLM checkpoint and **contains no
pooler at all**. Both "poolers" were random inits. Our published artifact therefore
ships 197 tensors *without* a pooler, so downstream pooler init is per-seed, same
as loading the official artifact would be.

Published: **https://huggingface.co/subrace/NepBERTa-pytorch** with a model card
recording source file, conversion date/toolchain, vocab md5
(`edfd394677436b306fb062159ec46c72`), and the bit-identity check.
`src/modeling.py` points at this checkpoint.

**Why:** Full provenance control. The report can now cite an unbroken chain:
official TF weights → our documented, verified conversion → the harness. No
dependence on an anonymous third-party artifact.

## 5. The LoRA pooler confound fix (`src/modeling.py`)

**What:** PEFT's `TaskType.SEQ_CLS` default keeps only the `classifier` trainable.
But BERT-family classification routes `[CLS]` through `bert.pooler` — which the
default leaves **frozen**. For NepBERTa (no trained pooler in the checkpoint) that
would mean LoRA training on top of a *frozen random* pooler: a confound that would
poison every NepBERTa×LoRA cell. Fix:

```python
modules_to_save=["classifier", "score", "pooler"]
```

**Why:** The pooler is functionally part of the classification head, so it should
be trainable like the head. RoBERTa-family models have no pooler, so the entry is a
no-op there. Verified consequence: **both families have exactly 887,042 trainable
params under LoRA (r=8, 2 labels)** — an equal-budget fairness claim the report
can make directly.

## 6. CPU smoke test (run by the user, 14 min)

**What:** `train.py --config configs/smoke_test.yaml` — XLM-R + LoRA on
coreference (564 train examples), 15 epochs, seed 42, CPU.

**Result — pipeline validated end-to-end:** data load, LoRA attach (887,042
trainable, confirming the pooler fix), training loop, val+test eval, adapter saved
to `adapters/xlmr-base_coreference_lora_r8_s42/`, one row appended to
`results/results.csv`.

**Caveat to watch:** the model did not learn — loss pinned at ~0.693 (= ln 2,
i.e. majority-class collapse on a binary task; val macro-F1 frozen at 0.3448).
This is plausibly a genuinely hard cell (LoRA + 564 examples), not a code bug.
Watch this cell on GPU; if it stays degenerate it's a *finding*, not a failure.

Housekeeping noted: delete this CPU row from `results/results.csv` before the real
campaign (the harness skips run_ids already present), and gitignore
`checkpoints/`, `adapters/`, `.venv/`, `smoke.log`.

## 7. Current status / next steps

- ✅ All four checkpoints verified; provenance settled.
- ✅ Harness validated end-to-end on CPU.
- ⏳ Rotate the HF write token used during upload (it appeared in a terminal session).
- ▶ On Colab: `scripts/make_configs.py` to generate the grid, then run the
  campaign (README's campaign order). Watch the XLM-R×LoRA×coreference cell.
