"""
Assemble the consumer-style 2x2 comparison figure for the synthetic healthcare study:
  rows    = ground truth  (correctly specified additive-smooth  /  misspecified non-additive)
  columns = training regime (unperturbed / perturbed +/-5%)
each panel overlays the four model classes (Convex only, Additive, Smooth, Additive+Smooth),
plotting out-of-sample Rel-L2_w vs training size. Mirrors the consumer figures
(smooth-additive-{,not-}perturbed.pdf + non-smooth-additive-{,not-}perturbed.pdf).

Reads the per-panel prediction CSVs written by run_panel.py from:
  correct-spec : MAIN_ROOT   (suffix "" and "_pert")
  misspecified : MISSPEC_ROOT (suffix "" and "_pert")
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PRED_OUT, PRED_CSV, DATA_ROOT

MODEL_PLOT = [
    ("nonadd_nonsmooth_lp", "Convex only",     "C0", "o"),
    ("add_lp",              "Additive",        "C2", "s"),
    ("nonadd_smooth",       "Smooth",          "C1", "D"),
    ("add_smooth",          "Additive+Smooth", "C3", "^"),
]
_STYLE = {"font.family": "serif", "font.size": 12, "axes.labelsize": 13,
          "axes.titlesize": 13, "legend.fontsize": 11, "mathtext.fontset": "cm"}


def _csv_path(root, model, suffix):
    return os.path.join(root, os.path.basename(PRED_OUT[model]) + suffix, PRED_CSV[model])


def _plot_panel(ax, root, suffix, title):
    for key, label, color, marker in MODEL_PLOT:
        p = _csv_path(root, key, suffix)
        if not os.path.exists(p):
            print(f"  missing {p}", flush=True); continue
        df = pd.read_csv(p).sort_values("training_size")
        ax.plot(df["training_size"], df["test_RelL2_w"], label=label, color=color,
                marker=marker, linewidth=1.8, markersize=6)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(True, ls=":", alpha=0.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-root", default=DATA_ROOT)
    ap.add_argument("--misspec-root", default=os.path.join(DATA_ROOT, "_misspec"))
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "consumer_style_2x2.png"))
    args = ap.parse_args()
    plt.rcParams.update(_STYLE)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    _plot_panel(axes[0, 0], args.main_root,    "",      "Correctly specified  —  Unperturbed")
    _plot_panel(axes[0, 1], args.main_root,    "_pert", "Correctly specified  —  Perturbed (±5%)")
    _plot_panel(axes[1, 0], args.misspec_root, "",      "Misspecified (non-additive)  —  Unperturbed")
    _plot_panel(axes[1, 1], args.misspec_root, "_pert", "Misspecified (non-additive)  —  Perturbed (±5%)")
    for ax in axes[1, :]:
        ax.set_xlabel(r"Training set size ($N$)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Rel-$L_2$ error of predicted beamlets")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True,
               edgecolor="black", bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
