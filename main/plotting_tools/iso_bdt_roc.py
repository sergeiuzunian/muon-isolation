#!/usr/bin/env python3

#iso_bdt_roc.py - Overlay of the Best Isolation-Cut ROC and the Best Physical-Cone BDT Holdout ROC
#rebuilds the BDT holdout split with the run's settings and scores it with the saved model; isolation is recomputed from the samples

#modules for file handling, model loading, numerical operations, roc curves, and plotting (Agg backend, no display on the batch nodes)
import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#muonBDT.py and isoPLOT.py are imported from their directories for the loaders, feature builder, split, and isolation
HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.dirname(HERE)
ROOT = os.path.dirname(MAIN)
sys.path.insert(0, os.path.join(MAIN, "muon_iso_BDT"))
sys.path.insert(0, os.path.join(MAIN, "muon_iso_cut"))
import muonBDT
import isoPLOT

#saved model of the best physical-cone BDT run (dR0.5, dz10, d0 both, K10, depth 3) and the output directory
BDT_MODEL = os.path.join(ROOT, "output",
                         "muonbdt_scan_dr0.5_dz10.0_K10_d0on_iso_rw_v1.0_nMAX_run1",
                         "cfg01_d3_n200_lr0.1_L1.0_mcw1.0", "model.pkl")
OUT = os.path.join(ROOT, "output", "iso_bdt_roc_overlay")

#settings matching that run; ISO_BDT_ROC_N in the environment overrides the event count for a quick test
NEVENTS = int(os.environ.get("ISO_BDT_ROC_N", "0"))
MIN_PT = 1000.0
BDT_DR = 0.5
BDT_DZ = 10.0
K = 10
ISO_DR = 0.4
SEED = 42
HOLD = 0.2


#function to rebuild the BDT holdout muons with the same split seed and score them with the saved model
def bdt_holdout_roc():
    df = muonBDT.load_sample(muonBDT.SIG_DEFAULT, NEVENTS)
    Xs, ys, es = muonBDT.build_muon_matrix(df, MIN_PT, BDT_DR, BDT_DZ, K,
                                           True, True, True, True, False, True, 1)
    del df
    df = muonBDT.load_sample(muonBDT.BKG_DEFAULT, NEVENTS)
    Xb, yb, eb = muonBDT.build_muon_matrix(df, MIN_PT, BDT_DR, BDT_DZ, K,
                                           True, True, True, True, False, False, 0)
    del df
    (_, _), (Xs_ho, ys_ho) = muonBDT.split_by_event(Xs, ys, es, HOLD, SEED)
    (_, _), (Xb_ho, yb_ho) = muonBDT.split_by_event(Xb, yb, eb, HOLD, SEED)
    X_ho = np.vstack([Xs_ho, Xb_ho])
    y_ho = np.concatenate([ys_ho, yb_ho])
    with open(BDT_MODEL, "rb") as fh:
        clf = pickle.load(fh)
    proba = clf.predict_proba(X_ho)[:, 1]
    fpr, tpr, _ = roc_curve(y_ho, proba)
    return fpr, tpr, float(auc(fpr, tpr))


#function to compute the isolation-cut roc on the full samples (no training, so no holdout)
def iso_full_roc():
    df = isoPLOT.load_sample(isoPLOT.SIG_DEFAULT, NEVENTS)
    iso_s, _ = isoPLOT.muon_isolations(df, MIN_PT, ISO_DR, "auto", 0.0, True)
    del df
    df = isoPLOT.load_sample(isoPLOT.BKG_DEFAULT, NEVENTS)
    iso_b, _ = isoPLOT.muon_isolations(df, MIN_PT, ISO_DR, "auto", 0.0, False)
    del df
    iso = np.concatenate([iso_s, iso_b])
    y = np.concatenate([np.ones(len(iso_s), dtype=int), np.zeros(len(iso_b), dtype=int)])
    fpr, tpr, _ = roc_curve(y, -iso)
    return fpr, tpr, float(auc(fpr, tpr))


#main function
#both roc curves on one plot
def main():
    print("building BDT holdout ROC (dR0.5, dz10, d0 both, K10)...")
    fb, tb, ab = bdt_holdout_roc()
    print("BDT holdout AUC = %.4f" % ab)
    print("building isolation ROC (dR0.4, dz auto)...")
    fi, ti, ai = iso_full_roc()
    print("isolation AUC = %.4f" % ai)
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fb, tb, color="steelblue", lw=2, label="BDT (holdout, AUC = %.4f)" % ab)
    ax.plot(fi, ti, color="firebrick", lw=2, label="Isolation Cut (AUC = %.4f)" % ai)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate (Non-Prompt Muons Misidentified as Prompt)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Prompt Muons Correctly Identified)", fontsize=11)
    ax.set_title("Prompt vs Non-Prompt Muon: BDT vs Isolation Cut", fontsize=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT, "iso_bdt_roc_overlay.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved " + path)


#run main() only when executed as a script, not when imported
if __name__ == "__main__":
    main()
