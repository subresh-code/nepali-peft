"""Analysis for the 78-run grid: summary tables + report figures.

Usage: python scripts/analyze.py   (writes results/summary.csv and results/figures/*.png)
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# validated categorical palette (dataviz skill, all-pairs pass on white)
MODEL_COLOR = {
    "nepberta": "#2a78d6",
    "iriis-roberta": "#eb6834",
    "mbert": "#1baf7a",
    "xlmr-base": "#4a3aa7",
}
METHOD_COLOR = {"full": "#2a78d6", "lora": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#c3c2b7", "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

df = pd.read_csv("results/results.csv")
df["lora_r"] = df["lora_r"].fillna(0).astype(int)

# ---- summary table: mean ± std over seeds -----------------------------------
g = df.groupby(["model", "task", "method", "lora_r"])
summary = g.agg(
    test_f1_mean=("test_macro_f1", "mean"), test_f1_std=("test_macro_f1", "std"),
    test_acc_mean=("test_accuracy", "mean"),
    trainable_params=("trainable_params", "first"),
    pct_trainable=("pct_trainable", "first"),
    vram_gb=("peak_vram_gb", "mean"), minutes=("wall_clock_min", "mean"),
    seeds=("seed", "count"),
).reset_index()
summary.to_csv("results/summary.csv", index=False)
print(summary.to_string(index=False))

# ---- LoRA vs full gap (r=8 only, per model x task) --------------------------
main = summary[(summary.method == "full") | (summary.lora_r == 8)]
piv = main.pivot_table(index=["model", "task"], columns="method",
                       values="test_f1_mean").reset_index()
piv["gap"] = piv["full"] - piv["lora"]
print("\nLoRA vs full gap (test macro-F1, mean over seeds):")
print(piv.to_string(index=False))

# ---- Fig 1: Pareto — test F1 vs trainable params, one panel per task --------
tasks = ["sentiment", "acceptability", "coreference"]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, task in zip(axes, tasks):
    sub = main[main.task == task]
    for _, r in sub.iterrows():
        ax.scatter(r.trainable_params, r.test_f1_mean,
                   c=MODEL_COLOR[r.model],
                   marker="o" if r.method == "full" else "^",
                   s=70, zorder=3, edgecolors="white", linewidths=1)
        ax.errorbar(r.trainable_params, r.test_f1_mean, yerr=r.test_f1_std,
                    c=MODEL_COLOR[r.model], lw=1.5, capsize=3, zorder=2)
    ax.set_xscale("log")
    ax.set_title(task, color=INK)
    ax.set_xlabel("trainable parameters")
axes[0].set_ylabel("test macro-F1")
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=m)
           for m, c in MODEL_COLOR.items()]
handles += [plt.Line2D([], [], marker="o", ls="", color=MUTED, label="full FT"),
            plt.Line2D([], [], marker="^", ls="", color=MUTED, label="LoRA r=8")]
fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
           bbox_to_anchor=(0.5, 1.12))
fig.savefig(FIG_DIR / "pareto.png")

# ---- Fig 2: LoRA rank ablation (xlmr-base coreference) ----------------------
abl = df[(df.model == "xlmr-base") & (df.task == "coreference") & (df.method == "lora")]
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(abl.lora_r, abl.test_macro_f1, c="#4a3aa7", s=70, zorder=3,
           edgecolors="white", linewidths=1)
mean = abl.groupby("lora_r").test_macro_f1.mean()
ax.plot(mean.index, mean.values, c="#4a3aa7", lw=2, zorder=2)
full_ref = df[(df.model == "xlmr-base") & (df.task == "coreference") &
              (df.method == "full")].test_macro_f1.mean()
ax.axhline(full_ref, color=MUTED, lw=1.5, ls="--")
ax.text(4, full_ref, "full FT", color=MUTED, va="bottom", ha="left")
ax.set_xscale("log", base=2)
ax.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
ax.set_xlabel("LoRA rank r")
ax.set_ylabel("test macro-F1")
ax.set_title("XLM-R coreference: rank ablation", color=INK)
fig.savefig(FIG_DIR / "rank_ablation.png")

# ---- Fig 3: efficiency — VRAM and wall-clock, full vs LoRA (sentiment) ------
eff = summary[(summary.task == "sentiment")]
models = list(MODEL_COLOR)
x = range(len(models))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, col, label in [(axes[0], "vram_gb", "peak VRAM (GB)"),
                       (axes[1], "minutes", "wall-clock (min)")]:
    for i, meth in enumerate(["full", "lora"]):
        vals = [eff[(eff.model == m) & (eff.method == meth)][col].iloc[0]
                for m in models]
        ax.bar([xi + i * 0.4 for xi in x], vals, width=0.38,
               color=METHOD_COLOR[meth], label="full FT" if meth == "full" else "LoRA r=8")
        for xi, v in zip(x, vals):
            ax.text(xi + i * 0.4, v, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8, color=INK)
    ax.set_xticks([xi + 0.2 for xi in x], models, rotation=20)
    ax.set_ylabel(label)
    ax.grid(axis="x", visible=False)
axes[0].legend(frameon=False)
fig.suptitle("Training cost on sentiment (65k examples)", color=INK)
fig.savefig(FIG_DIR / "efficiency.png")

print(f"\nFigures written to {FIG_DIR}/")
