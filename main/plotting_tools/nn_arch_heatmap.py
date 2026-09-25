#!/usr/bin/env python3

#nn_arch_heatmap.py - NN Holdout AUC Heatmap Over Hidden Layer Widths
#reads summary.csv of the first muonnn h*x* run of each architecture (no _runN suffix) and plots holdout auc vs (h1, h2)

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
#collect (h1, h2, holdout auc) from each run, fill the matrix, and draw the annotated heatmap
def main():
    rows = []
    for p in glob.glob(os.path.join(OUTPUT, "muonnn_*_h*x*_v1.0_nMAX", "summary.csv")):
        s = read_summary(p)
        try:
            rows.append((int(s["hidden1"]), int(s["hidden2"]), float(s["auc_holdout"])))
        except (KeyError, ValueError):
            continue
    if not rows:
        print("no NN architecture-scan runs found")
        return

    h1s = sorted({r[0] for r in rows})
    h2s = sorted({r[1] for r in rows})
    M = np.full((len(h1s), len(h2s)), np.nan)
    for a, b, auc in rows:
        M[h1s.index(a), h2s.index(b)] = auc

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(h2s)))
    ax.set_xticklabels(h2s)
    ax.set_yticks(range(len(h1s)))
    ax.set_yticklabels(h1s)
    ax.set_xlabel("Second Hidden Layer Width")
    ax.set_ylabel("First Hidden Layer Width")
    ax.set_title("NN Holdout AUC vs Hidden Layer Widths\n(dR0.5, dz10, K10, d0 both, lr=1e-3, 12 epochs)")

    lo, hi = np.nanmin(M), np.nanmax(M)
    for i in range(len(h1s)):
        for j in range(len(h2s)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.4f}", ha="center", va="center",
                        color="black" if M[i, j] > lo + 0.5 * (hi - lo) else "white", fontsize=10)

    fig.colorbar(im, ax=ax, label="Holdout AUC")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "nn_arch_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved " + path)


#run main() only when executed as a script, not when imported
if __name__ == "__main__":
    main()
