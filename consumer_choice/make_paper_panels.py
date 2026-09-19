"""Paper panels (Figs 1-2) from the completed R=20 reshuffle run.

Layout, colours, markers, axes and legend are the production plot_panel/plot_legend from
consumer_inverse_optimization.py, copied verbatim except for two deliberate differences:
  * the band is +-2 standard errors (the caption at tex L439/469/500 says "two standard errors";
    production plot_panel draws +-1 SE), and
  * the standard deviation uses ddof=1 (sample SE), not numpy's default ddof=0.
Aggregation is nanmean/nanstd with a per-point replication count, so cells missing from the
checkpoint (the 7 deterministic Stage-3 failures) simply reduce n at that point.

usage: python make_paper_panels_2se.py RES_DIR OUT_DIR [--exclude PATH]
  --exclude  a text file of cells to drop, one per line: utility,regime,size,rep,model
"""
import os, sys, argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
CC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CC)
import consumer_inverse_optimization as C

BAND_SE = 2.0

def plot_panel_2se(sizes, errs, out_path, title=""):
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.asarray(sizes, float)
    for mdl in C.MODELS:
        ys = np.atleast_2d(np.asarray(errs[mdl.name], float))
        mean = np.nanmean(ys, axis=0)
        mask = ~np.isnan(mean)
        if not mask.any():
            continue
        if ys.shape[0] > 1:
            n_ok = np.sum(~np.isnan(ys), axis=0)
            se = np.nanstd(ys, axis=0, ddof=1) / np.sqrt(np.maximum(n_ok, 1))
            ax.fill_between(xs[mask], (mean - BAND_SE * se)[mask], (mean + BAND_SE * se)[mask],
                            color=mdl.color, alpha=0.18, linewidth=0)
        ax.plot(xs[mask], mean[mask], marker=mdl.marker, color=mdl.color,
                linewidth=1.8, label=mdl.name)
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel(r"Test relative $L_2$ error")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.6)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

ap = argparse.ArgumentParser()
ap.add_argument("res"); ap.add_argument("out"); ap.add_argument("--exclude", default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
ck = pd.read_csv(os.path.join(a.res, "_cells_checkpoint.csv"))
n0 = len(ck)
if a.exclude:
    ex = set()
    for line in open(a.exclude, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        u, g, s, rp, m = [t.strip() for t in line.split(",")]
        ex.add((u, g, int(s), int(rp), m))
    keep = [not ((r.utility, r.regime, int(r["size"]), int(r.rep), r.model) in ex) for _, r in ck.iterrows()]
    ck = ck[keep]
    print("excluded %d cells (%d listed)" % (n0 - len(ck), len(ex)))
sizes = list(C.CFG.training_sizes)
reps = sorted(ck["rep"].unique())
print("sizes:", sizes, " reps:", len(reps), " rows:", len(ck), " band: +-%.0f SE (ddof=1)" % BAND_SE)
for (u, regime), fname in C.PANELS.items():
    if u not in set(ck.utility):
        continue
    errs = {}
    for m in C.MODELS:
        e = np.full((len(reps), len(sizes)), np.nan)
        g = ck[(ck.utility == u) & (ck.regime == regime) & (ck.model == m.name)]
        for _, r in g.iterrows():
            e[reps.index(r["rep"]), sizes.index(int(r["size"]))] = r["RelL2"]
        errs[m.name] = e
        miss = np.isnan(e).sum(axis=0)
        if miss.any():
            print("  n<20 %-8s %-14s %-16s" % (u, regime, m.name),
                  " ".join("N%d:%d" % (s, 20 - k) for s, k in zip(sizes, miss) if k))
    plot_panel_2se(sizes, errs, os.path.join(a.out, fname))
    print("wrote", os.path.join(a.out, fname))
C.plot_legend(os.path.join(a.out, "legend.pdf"))
print("wrote", os.path.join(a.out, "legend.pdf"))
