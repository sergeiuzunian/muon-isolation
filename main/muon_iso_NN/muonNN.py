#!/usr/bin/env python3

#muonNN.py - Prompt vs Non-Prompt Muon NN Classifier
#pytorch MLP on the same feature matrix as muonBDT.py, zero-padded and standardized instead of NaN-padded

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
from sklearn.metrics import roc_curve, auc, roc_auc_score

#modules for particle physics data handling and background muon (pT, eta) reweighting
import uproot
from hep_ml.reweight import GBReweighter

#paths for signal and background samples; can be overridden with --signal-path and --bkg-path
SIG_DEFAULT = "/lstr/sahara/niueftracking/kesedlac/muon_isolation/OUTPUT/run_InvPtPU200/"
BKG_DEFAULT = "/lstr/sahara/niueftracking/kesedlac/muon_isolation/OUTPUT/run_bjet/"

#muonNN.py version for keeping track of run data relative to script edits
VERSION = "1.0"

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

#function to build the feature matrix, one row per candidate muon (label 1=signal, 0=background)
#zero-padded version of the muonBDT.py builder, with the same feature switches
def build_muon_matrix(df, min_pt, dr_cut, dz_cut, K,
                      use_neighbors, use_muon_d0, use_nbr_d0, use_isolation, use_zeta,
                      require_single_muon, label):

    #column book-keeping for the feature matrix layout
    n_per_slot = 5 if use_nbr_d0 else 4
    n_base = 3 + (1 if use_muon_d0 else 0) + (1 if use_isolation else 0)
    n_orderings = 2 + (1 if use_zeta else 0)
    n_cols = n_base + (n_orderings * K * n_per_slot if use_neighbors else 0)
    need_cone = use_neighbors or use_isolation
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
        sub = dRc = dz = None
        if need_cone:
            dRc = compute_deltaR_rect(eta_mu, phi_mu, eta, phi)
            dz = np.abs(z0s_mu[:, None] - z0s[None, :])
            sub = (dRc < dr_cut) & (dz < dz_cut) & usable[None, :]
            sub[np.arange(cand.size), cand] = False

        #feature rows for this event, zero-padded
        chunk = np.zeros((cand.size, n_cols), dtype=np.float32)
        c = 0
        chunk[:, c] = pt_mu;          c += 1
        chunk[:, c] = eta_mu;         c += 1
        chunk[:, c] = np.abs(z0s_mu); c += 1
        if use_muon_d0:
            chunk[:, c] = d0[cand];   c += 1

        #relative track isolation = sum of cone track pT / muon pT
        if use_isolation:
            chunk[:, c] = (pt[None, :] * sub).sum(axis=1) / pt_mu; c += 1

        #neighbor slots per ordering: dR ascending, pT descending, optional zeta ascending
        if use_neighbors and K > 0:
            order_dr = np.argsort(np.where(sub, dRc, np.inf), axis=1, kind="stable")[:, :K]
            valid_dr = np.take_along_axis(sub, order_dr, axis=1)
            pt_row = np.broadcast_to(pt[None, :], sub.shape)
            order_pt = np.argsort(-np.where(sub, pt_row, -np.inf), axis=1, kind="stable")[:, :K]
            valid_pt = np.take_along_axis(sub, order_pt, axis=1)
            groups = [(n_base, order_dr, valid_dr),
                      (n_base + K * n_per_slot, order_pt, valid_pt)]
            if use_zeta:
                #zeta = sqrt((20*deltaR)^2 + dz^2), combined angular and longitudinal distance
                zeta = np.sqrt((dRc * 20.0) ** 2 + dz ** 2)
                order_z = np.argsort(np.where(sub, zeta, np.inf), axis=1, kind="stable")[:, :K]
                valid_z = np.take_along_axis(sub, order_z, axis=1)
                groups.append((n_base + 2 * K * n_per_slot, order_z, valid_z))

            for base, order, valid in groups:
                pt_g = pt[order].astype(np.float32)
                deta_g = (eta[order] - eta_mu[:, None]).astype(np.float32)
                dr_g = np.take_along_axis(dRc, order, axis=1).astype(np.float32)
                dz_g = (z0s[order] - z0s_mu[:, None]).astype(np.float32)
                gathered = [pt_g, deta_g, dr_g, dz_g]
                if use_nbr_d0:
                    gathered = [d0[order].astype(np.float32)] + gathered

                #zero out the slots that are not real cone tracks (after the relative subtractions)
                for g in gathered:
                    g[~valid] = 0.0
                end = base + K * n_per_slot
                for fo, g in enumerate(gathered):
                    chunk[:, base + fo:end:n_per_slot] = g

        Xc.append(chunk)
        yc.append(np.full(cand.size, label, dtype=np.int64))
        ec.append(np.full(cand.size, int(eid), dtype=np.int64))

    if not Xc:
        return np.empty((0, n_cols), np.float32), np.empty(0, np.int64), np.empty(0, np.int64)
    return np.vstack(Xc), np.concatenate(yc), np.concatenate(ec)

#function to compute per-muon weights making background (pT, eta) match signal; copied from muonBDT.py
def compute_gb_weights(pt, eta, y, n_estimators, max_depth, learning_rate, min_samples_leaf, clip):
    sig, bkg = y == 1, y == 0
    sig_feats = np.column_stack([pt[sig], eta[sig]])
    bkg_feats = np.column_stack([pt[bkg], eta[bkg]])
    print(f"  fitting GBReweighter: n_sig={sig.sum()}, n_bkg={bkg.sum()}")
    rw = GBReweighter(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                      min_samples_leaf=min_samples_leaf, gb_args={"subsample": 0.6})
    rw.fit(original=bkg_feats, target=sig_feats)

    #clip extreme weights (no single event dominates)
    #normalize background weights to mean 1
    bkg_w = np.clip(rw.predict_weights(bkg_feats), 1.0 / clip, clip)
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


#function to stamp run settings information box at the bottom of plots; copied from muonBDT.py
def add_settings_box(fig, settings_str):
    fig.text(0.01, -0.03, settings_str, ha="left", va="top", fontsize=8, family="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="whitesmoke", edgecolor="gray"))


#function to parse command line arguments for sample paths, cone cuts, and training settings
def parse_args():
    p = argparse.ArgumentParser(description="Prompt vs Non-Prompt Muon NN Classifier")
    p.add_argument("--signal-path", type=str, default=SIG_DEFAULT)
    p.add_argument("--bkg-path", type=str, default=BKG_DEFAULT)
    p.add_argument("--n-events", type=int, default=2000,
                   help="events to load from each sample; <=0 loads all events in the file")
    
    #pT cutoff selection and neighbor cone definition, same as muonBDT.py
    p.add_argument("--min-pt", type=float, default=1000.0)
    p.add_argument("--neighbor-dr-cut", type=float, default=0.5)
    p.add_argument("--neighbor-dz-cut", type=float, default=15.0,
                   help="|z0sinTheta_neighbor - z0sinTheta_muon| cut in mm")
    p.add_argument("--max-neighbors", type=int, default=10)

    #feature block selection, same as muonBDT.py
    p.add_argument("--use-neighbors", type=str, default="true", choices=["true", "false"])
    p.add_argument("--use-zeta-order", type=str, default="false", choices=["true", "false"])
    p.add_argument("--d0-mode", type=str, default="both", choices=["both", "muon-off", "none"])
    p.add_argument("--use-isolation", type=str, default="true", choices=["true", "false"])

    #evaluation protocol settings
    p.add_argument("--holdout-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)

    #network training settings
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--hidden1", type=int, default=20, help="first hidden layer width")
    p.add_argument("--hidden2", type=int, default=20, help="second hidden layer width")

    #choose if the GBReweighter should be used to reweight background (pT, eta) to match signal
    p.add_argument("--use-gbreweighter", type=str, default="true", choices=["true", "false"])

    #reweighting hyperparameters, same as muonBDT.py
    p.add_argument("--gbrw-n-estimators", type=int, default=80)
    p.add_argument("--gbrw-max-depth", type=int, default=3)
    p.add_argument("--gbrw-learning-rate", type=float, default=0.1)
    p.add_argument("--gbrw-min-samples-leaf", type=int, default=200)
    p.add_argument("--gbrw-clip", type=float, default=100.0)
    return p.parse_args()


#network architecture: n_in inputs -> h1 -> h2 -> 1 output; hidden widths set from the command line
class MLP(nn.Module):
    def __init__(self, n_in, h1, h2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, h1), nn.ReLU(),
                                 nn.Linear(h1, h2), nn.ReLU(),
                                 nn.Linear(h2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


#function to plot training and holdout loss versus epoch
def plot_loss_vs_epoch(train_losses, hold_losses, out_dir, settings_str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ep = range(1, len(train_losses) + 1)
    ax.plot(ep, train_losses, "o-", color="steelblue", lw=2, label="Training Loss (Weighted)")
    ax.plot(ep, hold_losses, "o-", color="darkorange", lw=2, label="Holdout Loss (Weighted)")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=12)
    ax.set_title("Training and Holdout Loss vs Epoch", fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout(); add_settings_box(fig, settings_str)
    plt.savefig(os.path.join(out_dir, "loss_vs_epoch.png"), dpi=150, bbox_inches="tight"); plt.close()


#function to plot holdout auc versus epoch with the best epoch marked
def plot_auc_vs_epoch(hold_aucs, out_dir, settings_str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ep = range(1, len(hold_aucs) + 1)
    best = int(np.argmax(hold_aucs))
    ax.plot(ep, hold_aucs, "o-", color="darkorange", lw=2, label="Holdout AUC")
    ax.scatter([best + 1], [hold_aucs[best]], color="darkorange", s=160, marker="*",
               edgecolors="black", zorder=5,
               label=f"Best (AUC = {hold_aucs[best]:.4f}, epoch {best + 1})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Holdout AUC", fontsize=12)
    ax.set_title("Holdout AUC vs Epoch", fontsize=12)
    ax.legend(fontsize=10, loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout(); add_settings_box(fig, settings_str)
    plt.savefig(os.path.join(out_dir, "auc_vs_epoch.png"), dpi=150, bbox_inches="tight"); plt.close()


#main function
#load and build both samples, split by event, reweight, standardize, then train the network and plot
def main():
    args = parse_args()
    torch.manual_seed(args.random_state)

    #convert string flags to booleans; d0-mode sets the muon and neighbor d0 switches
    use_neighbors = args.use_neighbors == "true"
    use_gbrw = args.use_gbreweighter == "true"
    d0_mode = args.d0_mode
    use_muon_d0 = d0_mode == "both"
    use_nbr_d0 = d0_mode in ("both", "muon-off")
    use_isolation = args.use_isolation == "true"
    use_zeta = args.use_zeta_order == "true"
    K = args.max_neighbors
    ev_tag = "MAX" if args.n_events <= 0 else str(args.n_events)

    #load and build the feature matrix for each sample
    #dataframe freed before loading the next
    print("loading + building signal (run_InvPtPU200)...")
    df = load_sample(args.signal_path, args.n_events)
    Xs, ys, es = build_muon_matrix(df, args.min_pt, args.neighbor_dr_cut, args.neighbor_dz_cut,
                                   K, use_neighbors, use_muon_d0, use_nbr_d0, use_isolation,
                                   use_zeta, require_single_muon=True, label=1)
    del df
    print("loading + building background (run_bjet)...")
    df = load_sample(args.bkg_path, args.n_events)
    Xb, yb, eb = build_muon_matrix(df, args.min_pt, args.neighbor_dr_cut, args.neighbor_dz_cut,
                                   K, use_neighbors, use_muon_d0, use_nbr_d0, use_isolation,
                                   use_zeta, require_single_muon=False, label=0)
    del df

    n_sig_ev = int(len(np.unique(es))); n_bkg_ev = int(len(np.unique(eb)))
    print(f"signal: {n_sig_ev:,} events -> {len(ys):,} muons | "
          f"background: {n_bkg_ev:,} events -> {len(yb):,} muons")

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
        w = compute_gb_weights(X_train[:, 0], X_train[:, 1], y_train, args.gbrw_n_estimators,
                               args.gbrw_max_depth, args.gbrw_learning_rate,
                               args.gbrw_min_samples_leaf, args.gbrw_clip)

    #class imbalance: upweight signal by n_bkg/n_sig on top of the reweighter weights
    spw = n_bkg_tr / max(n_sig_tr, 1)
    final_w = w.copy()
    final_w[y_train == 1] *= spw

    #standardize each feature with training statistics only
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    X_train = (X_train - mu) / sd
    X_hold = (X_hold - mu) / sd

    model = MLP(X_train.shape[1], args.hidden1, args.hidden2)
    n_params = sum(p.numel() for p in model.parameters())

    #run settings text for the plot settings boxes; _row pads labels so the box lines up as a table
    def _row(lbl, val): return f"{lbl:<14}: {val}"
    settings_data = "\n".join([
        f"RUN SETTINGS   (muonNN.py v{VERSION})",
        _row("Samples", "signal=run_InvPtPU200 (prompt), bkg=run_bjet (non-prompt); label=provenance"),
        _row("Candidates", f"muons (isMuon), pT >= {args.min_pt/1000:.1f} GeV; bkg all muons/event, signal 1/event"),
        _row("Features", f"muon pT, eta, |z0sinθ|" + (", d0 (signed)" if use_muon_d0 else " (muon d0 off)")
                         + (", isolation" if use_isolation else "")
                         + "; neighbors carry " + ("d0, " if use_nbr_d0 else "") + "pt, Δeta, ΔR, Δz0sinθ"
                         + f"  [d0-mode={d0_mode}]"),
        _row("Isolation", (f"ΣpT(cone tracks)/pT(muon); cone = dR<{args.neighbor_dr_cut} and "
                           f"|z0sinθ_track-z0sinθ_muon|<{args.neighbor_dz_cut}mm, pT>=1 GeV, muon excluded")
                          if use_isolation else "off"),
        _row("Neighbors", (f"<=K={K} same-event tracks, dR<{args.neighbor_dr_cut} and "
                           f"|Δz0sinθ|<{args.neighbor_dz_cut}mm (signed vertex sep); "
                           f"dR-asc, pT-desc" + (", ζ-asc" if use_zeta else "") + "; zero-padded")
                          if use_neighbors else "off"),
        _row("Network", f"MLP {X_train.shape[1]}->{args.hidden1}->{args.hidden2}->1, ReLU, {n_params} parameters"),
        _row("Training", f"Adam lr={args.learning_rate}, batch={args.batch_size}, epochs={args.epochs}; "
                         f"inputs standardized with training statistics"),
        _row("Reweighting", "GBReweighter (pT,eta) bkg->signal, train only" if use_gbrw else "off"),
        _row("Imbalance", f"scale_pos_weight={spw:.2f} on signal"),
        _row("Events", f"signal {n_sig_ev:,} events / {len(ys):,} muons, "
                       f"background {n_bkg_ev:,} events / {len(yb):,} muons"),
        _row("Train/holdout", f"{len(y_train):,} / {len(y_hold):,} muons (80/20 split by event)"),
    ])

    #per-sample weighted loss
    #reduction none keeps one loss per muon, multiplied by its weight
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    #training loop over minibatches
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                        torch.from_numpy(y_train.astype(np.float32)),
                                        torch.from_numpy(final_w))
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    #holdout tensors and weights for the per-epoch holdout loss
    #same class-imbalance weighting as training (spw on signal) so the two curves share a scale;
    #reweighter weights are train-only and not applied here
    Xh_t = torch.from_numpy(X_hold)
    yh_t = torch.from_numpy(y_hold.astype(np.float32))
    wh = np.ones_like(y_hold, dtype=np.float32)
    wh[y_hold == 1] *= spw
    wh_t = torch.from_numpy(wh)

    train_losses = []
    hold_losses = []
    hold_aucs = []
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for xb, yb_, wb in dl:
            opt.zero_grad()
            loss = (lossf(model(xb), yb_) * wb).mean()
            loss.backward()
            opt.step()
            total += float(loss) * len(yb_)
        train_losses.append(total / len(ds))
        #per-epoch holdout loss and holdout auc from the same forward pass
        model.eval()
        with torch.no_grad():
            hlogits = model(Xh_t)
            hold_losses.append(float((lossf(hlogits, yh_t) * wh_t).mean()))
            hold_aucs.append(float(roc_auc_score(y_hold, torch.sigmoid(hlogits).numpy())))
        model.train()
        print(f"epoch {epoch+1}/{args.epochs}  loss {train_losses[-1]:.4f} | "
              f"holdout loss {hold_losses[-1]:.4f} | holdout auc {hold_aucs[-1]:.4f}")

    #epoch with the best holdout auc
    best_ep = int(np.argmax(hold_aucs)) + 1
    auc_ho_best = float(hold_aucs[best_ep - 1])
    print(f"best holdout auc {auc_ho_best:.4f} at epoch {best_ep}/{args.epochs}")

    #evaluate scores for train and holdout, roc curves and auc
    model.eval()
    with torch.no_grad():
        proba_train = torch.sigmoid(model(torch.from_numpy(X_train))).numpy()
        proba_hold = torch.sigmoid(model(torch.from_numpy(X_hold))).numpy()
    fpr_tr, tpr_tr, th_tr = roc_curve(y_train, proba_train)
    fpr_ho, tpr_ho, th_ho = roc_curve(y_hold, proba_hold)
    auc_tr = float(auc(fpr_tr, tpr_tr))
    auc_ho = float(auc(fpr_ho, tpr_ho))
    print(f"auc train {auc_tr:.4f} | holdout {auc_ho:.4f} | gap {auc_tr-auc_ho:.4f}")

    #output directory; name encodes the cone, feature, and reweighting settings, same as muonBDT.py
    #suffix _runN avoids overwriting an existing one
    #project root is three levels up from this file (muonisolation/main/muon_iso_NN/)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output")
    nbr_str = f"dr{args.neighbor_dr_cut}_dz{args.neighbor_dz_cut}_K{K}" if use_neighbors else "noNbr"
    nbr_str += {"both": "_d0on", "muon-off": "_d0muOFF", "none": "_d0off"}[d0_mode]
    nbr_str += "_iso" if use_isolation else "_noiso"
    nbr_str += "_zeta" if use_zeta else ""
    nbr_str += "_rw" if use_gbrw else "_norw"
    base = os.path.join(output_dir, f"muonnn_{nbr_str}_h{args.hidden1}x{args.hidden2}_v{VERSION}_n{ev_tag}")
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

    #youden j operating points, same convention as muonBDT.py
    k_tr = int(np.argmax(tpr_tr - fpr_tr)); k_ho = int(np.argmax(tpr_ho - fpr_ho))
    ax.scatter([fpr_tr[k_tr]], [tpr_tr[k_tr]], color="steelblue", s=120, marker="o", edgecolors="black",
               zorder=5, label=f"Train Youden J (Score > {th_tr[k_tr]:.3f}, J = {tpr_tr[k_tr]-fpr_tr[k_tr]:.3f})")
    ax.scatter([fpr_ho[k_ho]], [tpr_ho[k_ho]], color="darkorange", s=160, marker="*", edgecolors="black",
               zorder=5, label=f"Holdout Youden J (Score > {th_ho[k_ho]:.3f}, J = {tpr_ho[k_ho]-fpr_ho[k_ho]:.3f})")
    ax.set_xlabel("False Positive Rate (Non-Prompt Muons Misidentified as Prompt)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Prompt Muons Correctly Identified)", fontsize=11)
    ax.set_title("Prompt vs Non-Prompt Muon NN Classifier - ROC", fontsize=12)
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); add_settings_box(fig, settings_data)
    plt.savefig(os.path.join(out, "roc.png"), dpi=150, bbox_inches="tight"); plt.close()

    #loss and auc vs epoch plots
    plot_loss_vs_epoch(train_losses, hold_losses, out, settings_data)
    plot_auc_vs_epoch(hold_aucs, out, settings_data)

    #saves the trained network and a summary of settings and results
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["key", "value"])
        for k, v in [("muonnn_version", VERSION), ("model_type", "mlp"),
                     ("n_events_requested", args.n_events), ("min_pt_MeV", args.min_pt),
                     ("neighbor_dr_cut", args.neighbor_dr_cut), ("neighbor_dz_cut_mm", args.neighbor_dz_cut),
                     ("max_neighbors_K", K), ("use_neighbors", use_neighbors),
                     ("use_zeta_order", use_zeta),
                     ("d0_mode", d0_mode), ("use_muon_d0", use_muon_d0), ("use_nbr_d0", use_nbr_d0),
                     ("use_isolation", use_isolation), ("use_gbreweighter", use_gbrw),
                     ("n_features", X_train.shape[1]),
                     ("hidden1", args.hidden1), ("hidden2", args.hidden2), ("n_params", n_params),
                     ("n_signal_events", n_sig_ev), ("n_bkg_events", n_bkg_ev),
                     ("n_signal_muons", len(ys)), ("n_bkg_muons", len(yb)),
                     ("n_train", len(y_train)), ("n_holdout", len(y_hold)), ("scale_pos_weight", spw),
                     ("epochs", args.epochs), ("batch_size", args.batch_size),
                     ("learning_rate", args.learning_rate),
                     ("auc_train", auc_tr), ("auc_holdout", auc_ho),
                     ("best_epoch", best_ep), ("auc_holdout_best", auc_ho_best)]:
            w.writerow([k, v])
    print("done")


#run main() only when executed as a script, not when imported
if __name__ == "__main__":
    main()
