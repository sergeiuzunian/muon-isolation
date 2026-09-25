#!/usr/bin/env python3

#nn_bdt_roc.py - Overlay of the Best BDT and Best NN Holdout ROC Curves
#both scored on the same holdout split (same seed) from their saved models

#modules for file handling, csv reading, model loading, numerical operations, the network, roc curves, and plotting (Agg backend)
import os
import sys
import csv
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#muonBDT.py and muonNN.py are imported from their directories for the loaders, builders, split, and network
HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.dirname(HERE)
ROOT = os.path.dirname(MAIN)
sys.path.insert(0, os.path.join(MAIN, "muon_iso_BDT"))
sys.path.insert(0, os.path.join(MAIN, "muon_iso_NN"))
import muonBDT
import muonNN

#best BDT run (depth 3), best NN run (15x15, run2), and the output directory
BDT_RUN = os.path.join(ROOT, "output",
                       "muonbdt_scan_dr0.5_dz10.0_K10_d0on_iso_rw_v1.0_nMAX_run1",
                       "cfg01_d3_n200_lr0.1_L1.0_mcw1.0")
NN_RUN = os.path.join(ROOT, "output",
                      "muonnn_dr0.5_dz10.0_K10_d0on_iso_rw_h15x15_v1.0_nMAX_run2")
OUT = os.path.join(ROOT, "output", "nn_bdt_roc_overlay")

#settings matching those runs; NN_BDT_ROC_N in the environment overrides the event count for a quick test
NEVENTS = int(os.environ.get("NN_BDT_ROC_N", "0"))
MIN_PT = 1000.0
DR = 0.5
DZ = 10.0
K = 10
SEED = 42
HOLD = 0.2


#function to read a run's summary.csv into a dictionary
def read_summary(path):
    d = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) == 2 and row[0] != "key":
                d[row[0]] = row[1]
    return d


#function to rebuild the train and holdout feature matrices with a module's builder (muonBDT or muonNN)
#train features are kept for the NN standardization statistics
def build_split(module):
    df = module.load_sample(module.SIG_DEFAULT, NEVENTS)
    Xs, ys, es = module.build_muon_matrix(df, MIN_PT, DR, DZ, K,
                                          True, True, True, True, False, True, 1)
    del df
    df = module.load_sample(module.BKG_DEFAULT, NEVENTS)
    Xb, yb, eb = module.build_muon_matrix(df, MIN_PT, DR, DZ, K,
                                          True, True, True, True, False, False, 0)
    del df
    (Xs_tr, ys_tr), (Xs_ho, ys_ho) = module.split_by_event(Xs, ys, es, HOLD, SEED)
    (Xb_tr, yb_tr), (Xb_ho, yb_ho) = module.split_by_event(Xb, yb, eb, HOLD, SEED)
    X_tr = np.vstack([Xs_tr, Xb_tr])
    X_ho = np.vstack([Xs_ho, Xb_ho])
    y_ho = np.concatenate([ys_ho, yb_ho])
    return X_tr, X_ho, y_ho


#function to score the BDT holdout with the saved model
def bdt_holdout_roc():
    _, X_ho, y_ho = build_split(muonBDT)
    with open(os.path.join(BDT_RUN, "model.pkl"), "rb") as fh:
        clf = pickle.load(fh)
    proba = clf.predict_proba(X_ho)[:, 1]
    fpr, tpr, _ = roc_curve(y_ho, proba)
    return fpr, tpr, float(auc(fpr, tpr))


#function to standardize the holdout with training statistics and score it with the saved network; also returns the run summary
def nn_holdout_roc():
    X_tr, X_ho, y_ho = build_split(muonNN)
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    X_ho = (X_ho - mu) / sd
    s = read_summary(os.path.join(NN_RUN, "summary.csv"))
    model = muonNN.MLP(X_ho.shape[1], int(s["hidden1"]), int(s["hidden2"]))
    model.load_state_dict(torch.load(os.path.join(NN_RUN, "model.pt"), map_location="cpu"))
    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(torch.from_numpy(X_ho.astype(np.float32)))).numpy()
    fpr, tpr, _ = roc_curve(y_ho, proba)
    return fpr, tpr, float(auc(fpr, tpr)), s


#main function
#both roc curves on one plot, with the run settings box
def main():
    print("building BDT holdout ROC (dR0.5, dz10, d0 both, K10)...")
    fb, tb, ab = bdt_holdout_roc()
    print("BDT holdout AUC = %.4f" % ab)
    print("building NN holdout ROC (same cone, zero-padded inputs)...")
    fn, tn, an, s = nn_holdout_roc()
    print("NN holdout AUC = %.4f" % an)
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fb, tb, color="steelblue", lw=2, label="BDT, depth 3 (holdout, AUC = %.4f)" % ab)
    ax.plot(fn, tn, color="seagreen", lw=2,
            label="NN, MLP %sx%s (holdout, AUC = %.4f)" % (s["hidden1"], s["hidden2"], an))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate (Non-Prompt Muons Misidentified as Prompt)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Prompt Muons Correctly Identified)", fontsize=11)
    ax.set_title("Prompt vs Non-Prompt Muon: BDT vs Neural Network", fontsize=12)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    settings = "\n".join([
        "RUN SETTINGS",
        f"{'Samples':<12}: signal=run_InvPtPU200 (prompt), bkg=run_bjet (non-prompt); label=provenance",
        f"{'Cone':<12}: dR<{DR} and |Δz0sinθ|<{DZ:g}mm, K={K} neighbors, d0-mode=both, isolation on",
        f"{'BDT':<12}: XGBoost trees=200, depth=3, lr=0.1; NaN-padded neighbor slots",
        f"{'NN':<12}: MLP {s['n_features']}->{s['hidden1']}->{s['hidden2']}->1, ReLU, "
        f"Adam lr={s['learning_rate']}, {s['epochs']} epochs; zero-padded, standardized",
        f"{'Reweighting':<12}: GBReweighter (pT,eta) bkg->signal, train only; scale_pos_weight on signal",
        f"{'Evaluation':<12}: same 80/20 split by event; curves from the 20% holdout muons",
    ])
    fig.text(0.01, -0.02, settings, ha="left", va="top", fontsize=8, family="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="whitesmoke", edgecolor="gray"))
    plt.tight_layout()
    path = os.path.join(OUT, "nn_bdt_roc_overlay.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved " + path)


#run main() only when executed as a script, not when imported
if __name__ == "__main__":
    main()
