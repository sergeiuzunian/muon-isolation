#!/usr/bin/env python3

#nn_roc_youden.py - Best NN ROC Regenerated With Youden J Markers
#rebuilds the split, standardizes with training statistics, and scores train and holdout with the saved run2 network

#modules for file handling, csv reading, numerical operations, the network, roc curves, and plotting (Agg backend)
import os
import sys
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#muonNN.py is imported from its directory for the loader, builder, split, network, and settings box
HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.dirname(HERE)
ROOT = os.path.dirname(MAIN)
sys.path.insert(0, os.path.join(MAIN, "muon_iso_NN"))
import muonNN

#best NN run (15x15, run2) and the output directory
NN_RUN = os.path.join(ROOT, "output",
                      "muonnn_dr0.5_dz10.0_K10_d0on_iso_rw_h15x15_v1.0_nMAX_run2")
OUT = os.path.join(ROOT, "output", "nn_roc_youden")

#settings matching that run; NN_ROC_N in the environment overrides the event count for a quick test
NEVENTS = int(os.environ.get("NN_ROC_N", "0"))
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


#main function
#rebuild and standardize the split, score with the saved network, roc with Youden J for train and holdout
def main():
    df = muonNN.load_sample(muonNN.SIG_DEFAULT, NEVENTS)
    Xs, ys, es = muonNN.build_muon_matrix(df, MIN_PT, DR, DZ, K,
                                          True, True, True, True, False, True, 1)
    del df
    df = muonNN.load_sample(muonNN.BKG_DEFAULT, NEVENTS)
    Xb, yb, eb = muonNN.build_muon_matrix(df, MIN_PT, DR, DZ, K,
                                          True, True, True, True, False, False, 0)
    del df
    (Xs_tr, ys_tr), (Xs_ho, ys_ho) = muonNN.split_by_event(Xs, ys, es, HOLD, SEED)
    (Xb_tr, yb_tr), (Xb_ho, yb_ho) = muonNN.split_by_event(Xb, yb, eb, HOLD, SEED)
    X_tr = np.vstack([Xs_tr, Xb_tr]); y_tr = np.concatenate([ys_tr, yb_tr])
    X_ho = np.vstack([Xs_ho, Xb_ho]); y_ho = np.concatenate([ys_ho, yb_ho])

    #standardize with training statistics only, same as muonNN.py
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    X_tr = (X_tr - mu) / sd
    X_ho = (X_ho - mu) / sd

    #load the saved network with the hidden widths from the run summary
    s = read_summary(os.path.join(NN_RUN, "summary.csv"))
    model = muonNN.MLP(X_tr.shape[1], int(s["hidden1"]), int(s["hidden2"]))
    model.load_state_dict(torch.load(os.path.join(NN_RUN, "model.pt"), map_location="cpu"))
    model.eval()
    with torch.no_grad():
        p_tr = torch.sigmoid(model(torch.from_numpy(X_tr.astype(np.float32)))).numpy()
        p_ho = torch.sigmoid(model(torch.from_numpy(X_ho.astype(np.float32)))).numpy()

    fpr_tr, tpr_tr, th_tr = roc_curve(y_tr, p_tr)
    fpr_ho, tpr_ho, th_ho = roc_curve(y_ho, p_ho)
    auc_tr = float(auc(fpr_tr, tpr_tr)); auc_ho = float(auc(fpr_ho, tpr_ho))
    k_tr = int(np.argmax(tpr_tr - fpr_tr)); k_ho = int(np.argmax(tpr_ho - fpr_ho))
    print("auc train %.4f | holdout %.4f" % (auc_tr, auc_ho))
    print("youden J train %.3f | holdout %.3f" % (tpr_tr[k_tr]-fpr_tr[k_tr], tpr_ho[k_ho]-fpr_ho[k_ho]))

    #run settings text for the plot settings box, filled from the run summary
    settings = "\n".join([
        "RUN SETTINGS   (muonNN.py v%s)" % s.get("muonnn_version", "1.0"),
        f"{'Samples':<14}: signal=run_InvPtPU200 (prompt), bkg=run_bjet (non-prompt); label=provenance",
        f"{'Candidates':<14}: muons (isMuon), pT >= 1.0 GeV; bkg all muons/event, signal 1/event",
        f"{'Features':<14}: muon pT, eta, |z0sinθ|, d0 (signed), isolation; neighbors carry d0, pt, Δeta, ΔR, Δz0sinθ  [d0-mode=both]",
        f"{'Neighbors':<14}: <=K={K} same-event tracks, dR<{DR} and |Δz0sinθ|<{DZ:g}mm; dR-asc, pT-desc; zero-padded",
        f"{'Network':<14}: MLP {s['n_features']}->{s['hidden1']}->{s['hidden2']}->1, ReLU, {s['n_params']} parameters",
        f"{'Training':<14}: Adam lr={s['learning_rate']}, batch={s['batch_size']}, epochs={s['epochs']}; inputs standardized with training statistics",
        f"{'Reweighting':<14}: GBReweighter (pT,eta) bkg->signal, train only",
        f"{'Train/holdout':<14}: {len(y_tr):,} / {len(y_ho):,} muons (80/20 split by event)",
    ])

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr_tr, tpr_tr, color="steelblue", lw=2, label=f"Train ROC (AUC = {auc_tr:.4f})")
    ax.plot(fpr_ho, tpr_ho, color="darkorange", lw=2, label=f"Holdout ROC (AUC = {auc_ho:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random")
    ax.scatter([fpr_tr[k_tr]], [tpr_tr[k_tr]], color="steelblue", s=120, marker="o", edgecolors="black",
               zorder=5, label=f"Train Youden J (Score > {th_tr[k_tr]:.3f}, J = {tpr_tr[k_tr]-fpr_tr[k_tr]:.3f})")
    ax.scatter([fpr_ho[k_ho]], [tpr_ho[k_ho]], color="darkorange", s=160, marker="*", edgecolors="black",
               zorder=5, label=f"Holdout Youden J (Score > {th_ho[k_ho]:.3f}, J = {tpr_ho[k_ho]-fpr_ho[k_ho]:.3f})")
    ax.set_xlabel("False Positive Rate (Non-Prompt Muons Misidentified as Prompt)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Prompt Muons Correctly Identified)", fontsize=11)
    ax.set_title("Prompt vs Non-Prompt Muon NN Classifier - ROC", fontsize=12)
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    muonNN.add_settings_box(fig, settings)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "nn_roc_best.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print("saved " + path)


#run main() only when executed as a script, not when imported
if __name__ == "__main__":
    main()
