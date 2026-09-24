"""Regenerate Fig 4 (combined beamlet-prediction error vs N) capped at N=200.

Data source: the CSVs extracted from the canonical additive-smooth repo at commit c589d16
("Switch prediction metric to Rel-L2") and frozen under figures/fig4_data/. Every point is the
procedure's own result (the Smooth N=20 point uses the bisection's beta* = 42.036 as returned).
Only the N=300,400 tail is dropped (the accuracy plateau sets in
by ~N=60, so the 200-400 stretch added width without information -- see Fig 3, now at
N=20,40,60). Convex-only = the approximate (max-forced) evaluator, matching the paper.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "fig4_data")
OUT = os.path.join(HERE, "combined_mape_vs_training_size_OR_style.png")

SIZES = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]   # capped at 200 (was ...,300,400)

# (csv, legend label, color, marker) -- colors/markers identical to plots.py MODEL_PLOT
MODEL_PLOT = [
    ("convex_only__nonadd_nonsmooth.csv", "Convex only",     "C0", "o"),
    ("additive__add_pure_lp.csv",         "Additive",        "C2", "s"),
    ("smooth__nonadd_smooth.csv",         "Smooth",          "C1", "D"),
    ("add_smooth__add_smooth.csv",        "Additive+Smooth", "C3", "^"),
]

_OR_STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 16,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
    "axes.grid": False, "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.5, "lines.markersize": 6, "axes.linewidth": 1.0, "mathtext.fontset": "cm",
}

plt.rcParams.update(_OR_STYLE)
plt.figure(figsize=(8, 6))
for fname, label, color, marker in MODEL_PLOT:
    df = pd.read_csv(os.path.join(DATA, fname))
    df = df[df["training_size"].isin(SIZES)].sort_values("training_size")
    plt.plot(df["training_size"], df["test_RelL2_w"], label=label, color=color,
             marker=marker, linestyle="-", linewidth=1.8)
plt.title("Beamlet Prediction Accuracy vs. Training Size")
plt.xlabel(r"Training Set Size ($N$)")
plt.ylabel(r"Relative $L_2$ error of predicted beamlets")
plt.xticks(SIZES)
plt.xlim(SIZES[0] - 8, SIZES[-1] + 8)
plt.legend(loc="upper right", frameon=True, edgecolor="black", fancybox=False, framealpha=1.0)
plt.grid(False); plt.tight_layout()
plt.savefig(OUT, dpi=300); plt.close()
print("wrote", OUT)
