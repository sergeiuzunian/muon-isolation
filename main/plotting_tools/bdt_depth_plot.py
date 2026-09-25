#!/usr/bin/env python3

#bdt_depth_plot.py - BDT Train and Holdout AUC vs Tree Depth
#reads scan_summary.csv of the dR0.5 dz10 d0on scans and plots the best holdout auc at each depth

#modules for file handling, csv reading, numerical operations, and plotting (Agg backend, no display on the batch nodes)
import os
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#project root is two levels up from this directory (muonisolation/main/plotting_tools/); plot goes to output/bdt_depth_scan
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUTPUT = os.path.join(ROOT, "output")
OUT = os.path.join(OUTPUT, "bdt_depth_scan")

#run settings text for the plot settings box; the scan settings are fixed by the directories matched in main
SETTINGS = "\n".join([
    "RUN SETTINGS",
    f"{'Samples':<12}: signal=run_InvPtPU200 (prompt), bkg=run_bjet (non-prompt); label=provenance",
    f"{'Cone':<12}: dR<0.5 and |Δz0sinθ|<10.0mm, K=10 neighbors, d0-mode=both, isolation on",
    f"{'Reweighting':<12}: GBReweighter (pT,eta) bkg->signal, train only; scale_pos_weight on signal",
    f"{'Selection':<12}: best holdout AUC per tree depth across dz10 d0on scans (full statistics)",
])


#main function
#best (train, holdout) auc per depth across the matching scans, then the plot
def main():
    best = {}
    for p in glob.glob(os.path.join(OUTPUT, "muonbdt_scan_dr0.5_dz10.0_K10_d0on_iso_rw_v*_nMAX*", "scan_summary.csv")):
        with open(p) as fh:
            for row in csv.DictReader(fh):
                d = int(float(row["max_depth"]))
                tr = float(row["auc_train"])
                ho = float(row["auc_holdout"])
                if d not in best or ho > best[d][1]:
                    best[d] = (tr, ho)
    if not best:
        print("no matching BDT depth-scan runs found")
        return

    depths = sorted(best)
    tr = [best[d][0] for d in depths]
    ho = [best[d][1] for d in depths]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot(depths, tr, "o-", color="steelblue", lw=2, label="Train AUC")
    ax.plot(depths, ho, "o-", color="darkorange", lw=2, label="Holdout AUC")
    ax.fill_between(depths, ho, tr, color="gray", alpha=0.15, label="Train-Holdout Gap")
    for d in depths:
        ax.annotate(f"gap {best[d][0] - best[d][1]:.3f}", (d, (best[d][0] + best[d][1]) / 2),
                    ha="center", fontsize=8, color="dimgray")
    ax.set_xlabel("Max Tree Depth", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("BDT Train and Holdout AUC vs Tree Depth", fontsize=12)
    ax.set_xticks(depths)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.text(0.01, -0.03, SETTINGS, ha="left", va="top", fontsize=8, family="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="whitesmoke", edgecolor="gray"))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "bdt_depth_auc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved " + path)


#run main() only when executed as a script, not when imported
if __name__ == "__main__":
    main()
