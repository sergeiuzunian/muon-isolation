#!/usr/bin/env python3

#note: work in progress. Not yet fully tested/ optimized. 
#goals:
# - add run settings box, include network architecture and training settings
# - add loss vs epoch plot
# - add option to turn d0 off for muons/ neighboring tracks

#modules for command line argument parsing and file handling
import argparse
import os
import csv

#module for progress bar in terminal output
from tqdm import tqdm

#modules for vectorized numerical operations, awkward array handling, and plotting
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

#modules for the neural network
import torch
import torch.nn as nn

#modules for the event split and roc curves
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

#modules for particle physics data handling and background muon (pT, eta) reweighting
import uproot
from hep_ml.reweight import GBReweighter

#paths for signal and background samples; can be overridden with --signal-path and --bkg-path
SIG_DEFAULT = "/lstr/sahara/niueftracking/kesedlac/muon_isolation/OUTPUT/run_InvPtPU200/"
BKG_DEFAULT = "/lstr/sahara/niueftracking/kesedlac/muon_isolation/OUTPUT/run_bjet/"

#muonNN.py version for keeping track of run data relative to script edits
VERSION = "1.0"

#number of neighbor variables per slot (d0, pt, deta, deltaR, dz0sintheta) and orderings (dR, pT)
N_PER_SLOT = 5
N_ORDERINGS = 2

#function to compute deltaR between muons and other tracks; copied from muonBDT.py
def compute_deltaR_rect(eta_mu, phi_mu, eta, phi):
    d_eta = eta_mu[:, None] - eta[None, :]
    #wrap-around for phi differences to ensure they are in the range [-pi, pi]
    d_phi = np.arctan2(np.sin(phi_mu[:, None] - phi[None, :]),
                       np.cos(phi_mu[:, None] - phi[None, :]))
    return np.sqrt(d_eta ** 2 + d_phi ** 2)


#function to load track branches from a sample's root file into a dataframe
#copied here from muonBDT.py
def load_sample(path, n_events):
    f = uproot.open(path + "OutputIsolation.root:OutputIsolation")
    stop = None if n_events <= 0 else n_events
    a = f.arrays(["InDetTrack_eta", "InDetTrack_phi", "InDetTrack_pt",
                  "InDetTrack_z0sinTheta", "InDetTrack_d0", "isMuon"], entry_stop=stop)
    return ak.to_dataframe(a)

#function to build the feature matrix, one row per candidate muon
#for now simplified copy of the one in muonBDT.py with
#default feature set fixed (muon pt, eta, |z0sinTheta|, d0, isolation and
#K neighbor slots per ordering)
#uses zeros instead of nan for empty slots
def build_muon_matrix(df, min_pt, dr_cut, dz_cut, K, require_single_muon, label):
    n_cols = 5 + N_ORDERINGS * K * N_PER_SLOT
    Xc, yc, ec = [], [], []
    for eid, ev in tqdm(df.groupby(level="entry"), desc=f"build label={label}"):
        eta = ev["InDetTrack_eta"].values
        phi = ev["InDetTrack_phi"].values
        pt = ev["InDetTrack_pt"].values
        z0s = ev["InDetTrack_z0sinTheta"].values
        d0 = ev["InDetTrack_d0"].values
        ismu = ev["isMuon"].values == True

        #skip signal events without exactly one muon; candidates are muons above the pT floor
        if require_single_muon and ismu.sum() != 1:
            continue
        usable = pt >= min_pt
        cand = np.where(ismu & usable)[0]
        if cand.size == 0:
            continue
        eta_mu, phi_mu, z0s_mu, pt_mu = eta[cand], phi[cand], z0s[cand], pt[cand]

        #cone mask (muons x tracks): deltaR cut and dz cut relative to the muon (signed z0sinTheta);
        #each muon's own track column is excluded
        dRc = compute_deltaR_rect(eta_mu, phi_mu, eta, phi)
        dz = np.abs(z0s_mu[:, None] - z0s[None, :])
        sub = (dRc < dr_cut) & (dz < dz_cut) & usable[None, :]
        sub[np.arange(cand.size), cand] = False

        #feature rows for this event, zero-padded
        chunk = np.zeros((cand.size, n_cols), dtype=np.float32)
        chunk[:, 0] = pt_mu
        chunk[:, 1] = eta_mu
        chunk[:, 2] = np.abs(z0s_mu)
        chunk[:, 3] = d0[cand]

        #relative track isolation = sum of cone track pT / muon pT
        chunk[:, 4] = (pt[None, :] * sub).sum(axis=1) / pt_mu

        #neighbor slots: K nearest cone tracks (dR ascending) and K hardest (pT descending)
        order_dr = np.argsort(np.where(sub, dRc, np.inf), axis=1, kind="stable")[:, :K]
        valid_dr = np.take_along_axis(sub, order_dr, axis=1)
        pt_row = np.broadcast_to(pt[None, :], sub.shape)
        order_pt = np.argsort(-np.where(sub, pt_row, -np.inf), axis=1, kind="stable")[:, :K]
        valid_pt = np.take_along_axis(sub, order_pt, axis=1)

        for base, order, valid in ((5, order_dr, valid_dr), (5 + K * N_PER_SLOT, order_pt, valid_pt)):
            d0_g = d0[order].astype(np.float32)
            pt_g = pt[order].astype(np.float32)
            deta_g = (eta[order] - eta_mu[:, None]).astype(np.float32)
            dr_g = np.take_along_axis(dRc, order, axis=1).astype(np.float32)
            dz_g = (z0s[order] - z0s_mu[:, None]).astype(np.float32)

            #zero out the slots that are not real cone tracks (after the relative subtractions)
            for g in (d0_g, pt_g, deta_g, dr_g, dz_g):
                g[~valid] = 0.0
            end = base + K * N_PER_SLOT
            chunk[:, base + 0:end:N_PER_SLOT] = d0_g
            chunk[:, base + 1:end:N_PER_SLOT] = pt_g
            chunk[:, base + 2:end:N_PER_SLOT] = deta_g
            chunk[:, base + 3:end:N_PER_SLOT] = dr_g
            chunk[:, base + 4:end:N_PER_SLOT] = dz_g

        Xc.append(chunk)
        yc.append(np.full(cand.size, label, dtype=np.int64))
        ec.append(np.full(cand.size, int(eid), dtype=np.int64))

    if not Xc:
        return np.empty((0, n_cols), np.float32), np.empty(0, np.int64), np.empty(0, np.int64)
    return np.vstack(Xc), np.concatenate(yc), np.concatenate(ec)

#function to compute per-muon weights making background (pT, eta) match signal; copied from
#muonBDT.py with the reweighter settings fixed (80 trees, depth 3, lr 0.1, min leaf 200, clip 100)
def compute_gb_weights(pt, eta, y):
    sig, bkg = y == 1, y == 0
    sig_feats = np.column_stack([pt[sig], eta[sig]])
    bkg_feats = np.column_stack([pt[bkg], eta[bkg]])
    print(f"  fitting GBReweighter: n_sig={sig.sum()}, n_bkg={bkg.sum()}")
    rw = GBReweighter(n_estimators=80, max_depth=3, learning_rate=0.1,
                      min_samples_leaf=200, gb_args={"subsample": 0.6})
    rw.fit(original=bkg_feats, target=sig_feats)

    #clip extreme weights (no single event dominates) 
    #normalize background weights to mean 1
    bkg_w = np.clip(rw.predict_weights(bkg_feats), 0.01, 100.0)
    w = np.ones_like(y, dtype=np.float32)
    w[bkg] = bkg_w
    w[bkg] /= w[bkg].mean()
    return w


#function to split muons into train/holdout by event id; copied from muonBDT.py
def split_by_event(X, y, eids, holdout_size, seed):
    ev = np.unique(eids)
    tr_ev, ho_ev = train_test_split(ev, test_size=holdout_size, random_state=seed)
    return (X[np.isin(eids, tr_ev)], y[np.isin(eids, tr_ev)]), \
           (X[np.isin(eids, ho_ev)], y[np.isin(eids, ho_ev)])


#function to parse command line arguments for sample paths, cone cuts, and training settings
def parse_args():
    p = argparse.ArgumentParser(description="Prompt vs Non-Prompt Muon NN Classifier")
    p.add_argument("--signal-path", type=str, default=SIG_DEFAULT)
    p.add_argument("--bkg-path", type=str, default=BKG_DEFAULT)
    p.add_argument("--n-events", type=int, default=2000,
                   help="events to load from each sample; <=0 loads all events in the file")
    p.add_argument("--min-pt", type=float, default=1000.0)
    p.add_argument("--neighbor-dr-cut", type=float, default=0.5)
    p.add_argument("--neighbor-dz-cut", type=float, default=15.0)
    p.add_argument("--max-neighbors", type=int, default=10)
    p.add_argument("--use-gbreweighter", type=str, default="true", choices=["true", "false"])
    p.add_argument("--holdout-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=0.001)
    return p.parse_args()


#network architecture
#105 inputs -> 20 -> 20 -> 1 output (~2.5k parameters, sized for ~20k events)
class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 20), nn.ReLU(),
                                 nn.Linear(20, 20), nn.ReLU(),
                                 nn.Linear(20, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    args = parse_args()
    torch.manual_seed(args.random_state)
    use_gbrw = args.use_gbreweighter == "true"
    K = args.max_neighbors
    ev_tag = "MAX" if args.n_events <= 0 else str(args.n_events)

    #load and build the feature matrix for each sample
    #dataframe freed before loading the next
    print("loading + building signal (run_InvPtPU200)...")
    df = load_sample(args.signal_path, args.n_events)
    Xs, ys, es = build_muon_matrix(df, args.min_pt, args.neighbor_dr_cut, args.neighbor_dz_cut,
                                   K, require_single_muon=True, label=1)
    del df
    print("loading + building background (run_bjet)...")
    df = load_sample(args.bkg_path, args.n_events)
    Xb, yb, eb = build_muon_matrix(df, args.min_pt, args.neighbor_dr_cut, args.neighbor_dz_cut,
                                   K, require_single_muon=False, label=0)
    del df

    #event-level train/holdout split for each sample, then combined
    (Xs_tr, ys_tr), (Xs_ho, ys_ho) = split_by_event(Xs, ys, es, args.holdout_size, args.random_state)
    (Xb_tr, yb_tr), (Xb_ho, yb_ho) = split_by_event(Xb, yb, eb, args.holdout_size, args.random_state)
    X_train = np.vstack([Xs_tr, Xb_tr]); y_train = np.concatenate([ys_tr, yb_tr])
    X_hold = np.vstack([Xs_ho, Xb_ho]); y_hold = np.concatenate([ys_ho, yb_ho])
    n_sig_tr = int((y_train == 1).sum()); n_bkg_tr = int((y_train == 0).sum())
    print(f"train: {len(y_train):,} muons ({n_sig_tr} sig, {n_bkg_tr} bkg) | holdout: {len(y_hold):,} muons")

    #background (pT, eta) reweighting, fit on the raw training columns before standardization
    w = np.ones_like(y_train, dtype=np.float32)
    if use_gbrw:
        w = compute_gb_weights(X_train[:, 0], X_train[:, 1], y_train)

    #class imbalance: upweight signal by n_bkg/n_sig on top of the reweighter weights
    final_w = w.copy()
    final_w[y_train == 1] *= n_bkg_tr / max(n_sig_tr, 1)

    #standardize each feature with training statistics only
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    X_train = (X_train - mu) / sd
    X_hold = (X_hold - mu) / sd

    model = MLP(X_train.shape[1])

    #per-sample weighted loss
    #reduction none keeps one loss per muon, multiplied by its weight
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    #training loop over minibatches
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                        torch.from_numpy(y_train.astype(np.float32)),
                                        torch.from_numpy(final_w))
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for xb, yb_, wb in dl:
            opt.zero_grad()
            loss = (lossf(model(xb), yb_) * wb).mean()
            loss.backward()
            opt.step()
            total += float(loss) * len(yb_)
        print(f"epoch {epoch+1}/{args.epochs}  loss {total/len(ds):.4f}")

    #evaluate scores for train and holdout, roc curves and auc
    model.eval()
    with torch.no_grad():
        proba_train = torch.sigmoid(model(torch.from_numpy(X_train))).numpy()
        proba_hold = torch.sigmoid(model(torch.from_numpy(X_hold))).numpy()
    fpr_tr, tpr_tr, _ = roc_curve(y_train, proba_train)
    fpr_ho, tpr_ho, _ = roc_curve(y_hold, proba_hold)
    auc_tr = float(auc(fpr_tr, tpr_tr))
    auc_ho = float(auc(fpr_ho, tpr_ho))
    print(f"auc train {auc_tr:.4f} | holdout {auc_ho:.4f} | gap {auc_tr-auc_ho:.4f}")

    #output directory; suffix _runN avoids overwriting an existing one
    #project root is three levels up from this file (muonisolation/main/muon_iso_NN/)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output")
    rw_tag = "_rw" if use_gbrw else "_norw"
    base = os.path.join(output_dir, f"muonnn_dr{args.neighbor_dr_cut}_dz{args.neighbor_dz_cut}_K{K}{rw_tag}_v{VERSION}_n{ev_tag}")
    out = base; i = 0
    while os.path.exists(out):
        i += 1; out = f"{base}_run{i}"
    os.makedirs(out)
    print(f"output dir: {out}")

    #roc plot plotting
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr_tr, tpr_tr, color="steelblue", lw=2, label=f"Train ROC (AUC = {auc_tr:.4f})")
    ax.plot(fpr_ho, tpr_ho, color="darkorange", lw=2, label=f"Holdout ROC (AUC = {auc_ho:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate (Non-Prompt Muons Misidentified as Prompt)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Prompt Muons Correctly Identified)", fontsize=11)
    ax.set_title("Prompt vs Non-Prompt Muon NN Classifier — ROC", fontsize=12)
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "roc.png"), dpi=150, bbox_inches="tight"); plt.close()

    #saves the trained network and a summary of settings and results
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["key", "value"])
        for k, v in [("muonnn_version", VERSION), ("model_type", "mlp"),
                     ("n_events_requested", args.n_events), ("min_pt_MeV", args.min_pt),
                     ("neighbor_dr_cut", args.neighbor_dr_cut), ("neighbor_dz_cut_mm", args.neighbor_dz_cut),
                     ("max_neighbors_K", K), ("use_gbreweighter", use_gbrw),
                     ("n_train", len(y_train)), ("n_holdout", len(y_hold)),
                     ("epochs", args.epochs), ("batch_size", args.batch_size),
                     ("learning_rate", args.learning_rate),
                     ("auc_train", auc_tr), ("auc_holdout", auc_ho)]:
            w.writerow([k, v])
    print("done")


if __name__ == "__main__":
    main()
