# Parameter-Efficient Adaptation for Nepali NLU — Experiment Harness

Implements the CMP6232 Assessment 2 experiments: a 2×2 factorial
(model family × adaptation method) over three publicly released NLUE
tasks, 3 seeds per condition, plus a LoRA rank ablation — with
accuracy AND efficiency (params, VRAM, wall-clock) recorded per run.

## Layout
```
train.py                 # one run = one config = one results row
src/data.py              # 3 NLUE tasks, fixed stratified 10% val split
src/modeling.py          # 4 encoders, full FT or LoRA (q/v, r configurable)
src/metrics.py           # Macro-F1, accuracy, MCC (acceptability)
scripts/verify_models.py # RUN FIRST: confirms checkpoint IDs + true params
scripts/make_configs.py  # writes all 79 grid configs
configs/smoke_test.yaml  # 5–10 min end-to-end sanity run
results/results.csv      # accumulates one row per completed run
```

## Quickstart (Colab, GPU runtime)
```bash
pip install -r requirements.txt -q
python scripts/verify_models.py        # step 1: settle checkpoint IDs
python train.py --config configs/smoke_test.yaml   # step 2: smoke test
python scripts/make_configs.py         # step 3: generate the grid
for f in configs/grid/*.yaml; do python train.py --config "$f"; done
```
Colab preempts: safe. Checkpointing resumes at the last epoch, and
completed runs already have their row in `results/results.csv` — skip
any config whose `run_id` is already present.

## Campaign order (matches the study plan)
1. Smoke test (coreference is tiny: 564 train examples).
2. Monolingual full FT (all tasks, 3 seeds) → compare against
   published numbers before proceeding.
3. Multilingual full FT.  4. All LoRA cells.  5. Rank ablation.
6. Analysis: Pareto plot (test macro-F1 vs trainable params), rank
   curves, per-class errors, tokenizer fertility.

## Protocol facts to restate in the report
- Validation is a stratified 10% hold-out (SPLIT_SEED=1234, identical
  across ALL runs); only training randomness varies with the run seed.
- Fixed budget across methods within a task (same epochs/batch/length);
  LR is per-method standard practice: 2e-5 full, 2e-4 LoRA.
- LoRA targets attention query/value projections (Hu et al., 2022).

## Integrity note
This harness is engineering scaffolding. The experiments you run, the
numbers you obtain, the analysis, and every word of the Assessment 2
report are yours — and you should be able to explain any line of this
code in a viva. Check your module's rules on code assistance and
declare AI involvement per BCU policy.
