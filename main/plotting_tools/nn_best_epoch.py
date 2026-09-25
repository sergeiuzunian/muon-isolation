#!/usr/bin/env python3

#nn_best_epoch.py - NN Best Holdout AUC and Best Epoch per Hidden Layer Configuration
#reads summary.csv of the muonnn runs that tracked auc per epoch and draws two bar panels sorted by best auc

#modules for file handling, csv reading, numerical operations, and plotting (Agg backend, no display on the batch nodes)
import os
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#project root is two levels up from this directory (muonisolation/main/plotting_tools/); plot goes to output/nn_arch_scan
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUTPUT = os.path.join(ROOT, "output")
OUT = os.path.join(OUTPUT, "nn_arch_scan")


#function to read a run's summary.csv into a dictionary
def read_summary(path):
    d = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) == 2 and row[0] != "key":
                d[row[0]] = row[1]
    return d


#main function
#collect the runs, sort by best holdout auc, then bar panels for auc and best epoch with the settings box
def main():
    rows = []
    for p in glob.glob(os.path.join(OUTPUT, "muonnn_*_h*x*_v1.0_nMAX*", "summary.csv")):
        s = read_summary(p)
        if "best_epoch" not in s:
            continue
        try:
            rows.append({"h1": int(s["hidden1"]), "h2": int(s["hidden2"]),
                         "params": int(s["n_params"]), "epoch": int(s["best_epoch"]),
                         "auc": float(s["auc_holdout_best"]), "max_ep": int(s["epochs"]),
                         "lr": float(s["learning_rate"])})
        except (KeyError, ValueError):
            continue
    if not rows:
        print("no NN runs with per-epoch AUC tracking found")
        return

    rows.sort(key=lambda r: -r["auc"])
    labels = [f"{r['h1']}x{r['h2']}" for r in rows]
    epochs = [r["epoch"] for r in rows]
    aucs = [r["auc"] for r in rows]
    x = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [1.15, 1]})

    bars = ax1.bar(x, aucs, color="steelblue", edgecolor="black")
    best = int(np.argmax(aucs))
    bars[best].set_color("darkorange")
    lo, hi = min(aucs), max(aucs)
    pad = (hi - lo) * 0.35 if hi > lo else 0.01
    ax1.set_ylim(lo - pad, hi + pad)
    for xi, a in zip(x, aucs):
        ax1.text(xi, a + (hi - lo) * 0.04, f"{a:.4f}", ha="center", fontsize=9)
    ax1.set_ylabel("Holdout AUC at Best Epoch", fontsize=11)
    ax1.set_title("NN Best Holdout AUC and Optimal Epoch by Hidden-Layer Configuration", fontsize=12)
    ax1.grid(alpha=0.3, axis="y")

    ax2.bar(x, epochs, color="seagreen", edgecolor="black")
    for xi, e in zip(x, epochs):
        ax2.text(xi, e + max(epochs) * 0.02, str(e), ha="center", fontsize=9)
    ax2.set_ylim(0, max(epochs) * 1.18)
    ax2.set_ylabel("Best Epoch", fontsize=11)
    ax2.set_xlabel("Hidden Layer Widths (h1 x h2)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{l}\n{r['params']}p" for l, r in zip(labels, rows)], fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    settings = "\n".join([
        "RUN SETTINGS   (muonNN.py)",
        f"{'Samples':<12}: signal=run_InvPtPU200 (prompt), bkg=run_bjet (non-prompt); label=provenance",
        f"{'Cone':<12}: dR<0.5 and |Δz0sinθ|<10.0mm, K=10 neighbors, d0-mode=both, isolation on",
        f"{'Training':<12}: Adam lr={rows[0]['lr']}, batch=512, up to {rows[0]['max_ep']} epochs; inputs standardized (train stats)",
        f"{'Reweighting':<12}: GBReweighter (pT,eta) bkg->signal, train only; scale_pos_weight on signal",
        f"{'Best epoch':<12}: epoch of maximum holdout AUC (selected on holdout)",
    ])
    fig.tight_layout()
    fig.text(0.01, -0.02, settings, ha="left", va="top", fontsize=8, family="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="whitesmoke", edgecolor="gray"))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "nn_best_epoch.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved " + path)


#run main() only when executed as a script, not when imported
if __name__ == "__main__":
    main()
