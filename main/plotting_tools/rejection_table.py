#!/usr/bin/env python3

#rejection_table.py - Non-Prompt Rejection at Fixed Prompt Efficiency for BDT, NN, and Isolation Cut
#rejection = 1/FPR read off each roc curve at the target TPR values in EFFS; writes rejection_table.csv

#modules for file handling, csv writing, model loading, numerical operations, the network, and roc curves
import os
import sys
import csv
import pickle
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc

#muonBDT.py, muonNN.py, and isoPLOT.py are imported from their directories for the loaders, builders, split, network, and isolation
HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.dirname(HERE)
ROOT = os.path.dirname(MAIN)
sys.path.insert(0, os.path.join(MAIN, "muon_iso_BDT"))
sys.path.insert(0, os.path.join(MAIN, "muon_iso_NN"))
sys.path.insert(0, os.path.join(MAIN, "muon_iso_cut"))
import muonBDT
import muonNN
import isoPLOT

#best BDT run (depth 3), best NN run (15x15, run2), and the output directory
BDT_RUN = os.path.join(ROOT, "output",
                       "muonbdt_scan_dr0.5_dz10.0_K10_d0on_iso_rw_v1.0_nMAX_run1",
                       "cfg01_d3_n200_lr0.1_L1.0_mcw1.0")
NN_RUN = os.path.join(ROOT, "output",
                      "muonnn_dr0.5_dz10.0_K10_d0on_iso_rw_h15x15_v1.0_nMAX_run2")
OUT = os.path.join(ROOT, "output", "rejection_table")

#settings matching those runs; REJ_N in the environment overrides the event count for a quick test
NEVENTS = int(os.environ.get("REJ_N", "0"))
MIN_PT = 1000.0
DR = 0.5
DZ = 10.0
K = 10
ISO_DR = 0.4
SEED = 42
HOLD = 0.2

#prompt efficiencies (TPR) at which the rejection is read off
EFFS = [0.99, 0.95, 0.90, 0.80, 0.70, 0.55, 0.40]


#function to read a run's summary.csv into a dictionary
def read_summary(path):
    d = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) == 2 and row[0] != "key":
                d[row[0]] = row[1]
    return d


#function to rebuild the train and holdout feature matrices with a module's builder (muonBDT or muonNN)
def build_split(module):
    df = module.load_sample(module.SIG_DEFAULT, NEVENTS)
    Xs, ys, es = module.build_muon_matrix(df, MIN_PT, DR, DZ, K,
                                          True, True, True, True, False, True, 1)
    del df
    df = module.load_sample(module.BKG_DEFAULT, NEVENTS)
    Xb, yb, eb = module.build_muon_matrix(df, MIN_PT, DR, DZ, K,
                                          True, True, True, True, False, False, 0)
    del df
    (Xs_tr, _), (Xs_ho, ys_ho) = module.split_by_event(Xs, ys, es, HOLD, SEED)
    (Xb_tr, _), (Xb_ho, yb_ho) = module.split_by_event(Xb, yb, eb, HOLD, SEED)
    return (np.vstack([Xs_tr, Xb_tr]), np.vstack([Xs_ho, Xb_ho]),
            np.concatenate([ys_ho, yb_ho]))


#function to score the BDT holdout with the saved model; returns the roc curve
def bdt_roc():
    _, X_ho, y_ho = build_split(muonBDT)
    with open(os.path.join(BDT_RUN, "model.pkl"), "rb") as fh:
        clf = pickle.load(fh)
    return roc_curve(y_ho, clf.predict_proba(X_ho)[:, 1])


#function to standardize the holdout with training statistics and score it with the saved network; returns the roc curve
def nn_roc():
    X_tr, X_ho, y_ho = build_split(muonNN)
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    X_ho = (X_ho - mu) / sd
    s = read_summary(os.path.join(NN_RUN, "summary.csv"))
    model = muonNN.MLP(X_ho.shape[1], int(s["hidden1"]), int(s["hidden2"]))
    model.load_state_dict(torch.load(os.path.join(NN_RUN, "model.pt"), map_location="cpu"))
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(X_ho.astype(np.float32)))).numpy()
    return roc_curve(y_ho, p)


#function to compute the isolation-cut roc on the full samples
def iso_roc():
    df = isoPLOT.load_sample(isoPLOT.SIG_DEFAULT, NEVENTS)
    iso_s, _ = isoPLOT.muon_isolations(df, MIN_PT, ISO_DR, "auto", 0.0, True)
    del df
    df = isoPLOT.load_sample(isoPLOT.BKG_DEFAULT, NEVENTS)
    iso_b, _ = isoPLOT.muon_isolations(df, MIN_PT, ISO_DR, "auto", 0.0, False)
    del df
    y = np.concatenate([np.ones(len(iso_s), dtype=int), np.zeros(len(iso_b), dtype=int)])
    return roc_curve(y, -np.concatenate([iso_s, iso_b]))


#function to read 1/FPR off a roc curve at each efficiency in EFFS; inf when the FPR is zero
def rejections(fpr, tpr):
    out = []
    for e in EFFS:
        i = int(np.searchsorted(tpr, e))
        i = min(i, len(tpr) - 1)
        f = fpr[i]
        out.append(float("inf") if f <= 0 else 1.0 / f)
    return out


#main function
#auc and rejections for each classifier, printed and written to rejection_table.csv
def main():
    rows = []
    for name, fn in [("BDT", bdt_roc), ("NN", nn_roc), ("Isolation cut", iso_roc)]:
        print(f"computing {name}...")
        fpr, tpr, _ = fn()
        a = float(auc(fpr, tpr))
        rej = rejections(fpr, tpr)
        rows.append((name, a, rej))
        print(f"  {name}: AUC = {a:.4f}")
        for e, r in zip(EFFS, rej):
            print(f"    eff {e:.2f} -> rejection {r:.1f}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "rejection_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "auc"] + [f"rej_at_eff{e:g}" for e in EFFS])
        for name, a, rej in rows:
            w.writerow([name, f"{a:.4f}"] + [f"{r:.2f}" for r in rej])
    print("saved " + os.path.join(OUT, "rejection_table.csv"))


#run main() only when executed as a script, not when imported
if __name__ == "__main__":
    main()
