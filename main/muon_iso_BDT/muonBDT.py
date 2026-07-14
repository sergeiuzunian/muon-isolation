#!/usr/bin/env python3

#muonBDT.py - Prompt vs Non-Prompt Muon BDT Classifier
#each comment line refers to code in the following block, in order of appearance

#modules for command line argument parsing, file handling, and serialization
import argparse
import os
import csv
import pickle

#module for progress bar in terminal output
from tqdm import tqdm 

#modules for vectorized numerical operations, awkward array handling, and plotting
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt

#modules for machine learning and statistical analysis
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score

#modules for particle physics data handling and signal/background muon (pT, eta) reweighting 
import uproot
from hep_ml.reweight import GBReweighter

#Kolmogorov-Smirnov test for comparing signal/background (pT, eta) distributions before/after reweighting
from scipy.stats import ks_2samp

#function to compute deltaR between muons and other tracks (self pairs delt with in build_muon_matrix)
def compute_deltaR_rect(eta_mu, phi_mu, eta, phi):
    d_eta = eta_mu[:, None] - eta[None, :]
    #wrap-around for phi differences to ensure they are in the range [-pi, pi]
    d_phi = np.arctan2(np.sin(phi_mu[:, None] - phi[None, :]),
                       np.cos(phi_mu[:, None] - phi[None, :]))
    return np.sqrt(d_eta ** 2 + d_phi ** 2)

#paths for signal and background samples; can be overridden with --signal-path and --bkg-path
SIG_DEFAULT = "/lstr/sahara/niueftracking/kesedlac/muon_isolation/OUTPUT/run_InvPtPU200/"
BKG_DEFAULT = "/lstr/sahara/niueftracking/kesedlac/muon_isolation/OUTPUT/run_bjet/"

#muonBDT.py version for keeping track of run data relative to script edits
VERSION = "1.1"

#dictionaries for displaying feature names in plots and tables
BASE_DISPLAY = {"pt": "pT", "d0": "d0", "abs_z0sinTheta": "|z0sinθ|", "isolation": "isolation"}
NBH_PHYS_DISPLAY = {"d0": "d0", "dz0sintheta": "Δz0sinθ", "pt": "pT", "deta": "Δeta", "deltaR": "ΔR"}

#function to translate from internal feature names to display form
def display_name(name):
    if "_nbh_" in name:
        phys, rest = name.split("_nbh_", 1)
        return f"{NBH_PHYS_DISPLAY.get(phys, phys)}_nbh_{rest}"
    return BASE_DISPLAY.get(name, name)

#function to generate a display title for neighbor feature groups based on their ordering and physical feature
def nbh_group_title(group_key):
    phys, ordering = group_key.split("_nbh_")
    order_desc = {"dR": "ΔR-Ordered, Nearest First",
                  "pT": "pT-Ordered, Hardest First",
                  "zeta": "ζ-Ordered, Nearest First"}.get(ordering, ordering)
    return f"Neighbor {NBH_PHYS_DISPLAY.get(phys, phys)} [{order_desc}]"

#module constants for labeling signal and background in plots
SIG_LABEL = "Prompt Muon (Signal)"
BKG_LABEL = "Non-Prompt Muon (Background)"

#function to stamp run settings information (hyperparameters, etc.) box at the bottom of plots
def add_settings_box(fig, settings_str, extra_text=None):
    txt = settings_str if extra_text is None else extra_text + "\n\n" + settings_str
    fig.text(0.01, -0.03, txt, ha="left", va="top", fontsize=8, family="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="whitesmoke", edgecolor="gray"))

#function to parse command line arguments for sample paths, hyperparameters, and feature selection
def parse_args():
    p = argparse.ArgumentParser(description="Prompt vs Non-Prompt Muon BDT Classifier")

    #sample selection
    p.add_argument("--signal-path", type=str, default=SIG_DEFAULT)
    p.add_argument("--bkg-path", type=str, default=BKG_DEFAULT)

    #hyperparameter and feature selection

    #choose number of events for training/holdout 
    p.add_argument("--n-events", type=int, default=2000,
                   help="events to load from each sample; <=0 loads all events in the file")
    
    #build-determining hyperparameters for the feature matrix
    #these are not changed during a build-once scan

    #pT cutoff selection and neighbor cone definition; 
    #neighbor slots are filled with tracks within this cone
    p.add_argument("--min-pt", type=float, default=1000.0)
    p.add_argument("--neighbor-dr-cut", type=float, default=0.5)
    p.add_argument("--neighbor-dz-cut", type=float, default=15.0,
                   help="|z0sinTheta_neighbor - z0sinTheta_muon| cut in mm")
    p.add_argument("--max-neighbors", type=int, default=10)

    #feature block selection
    p.add_argument("--use-neighbors", type=str, default="true", choices=["true", "false"])
    p.add_argument("--use-zeta-order", type=str, default="false", choices=["true", "false"])
    p.add_argument("--d0-mode", type=str, default="both", choices=["both", "muon-off", "none"])
    p.add_argument("--use-isolation", type=str, default="true", choices=["true", "false"])

    #evaluation protocol settings
    p.add_argument("--holdout-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    
    #hyperparameter override strings; can be repeated for a build-once scan
    p.add_argument("--config", type=str, action="append", default=None,
                   help="hyperparameter override string; repeat for a build-once scan")
    
    #xgboost hyperparameters; can be overridden with --config key=value,... strings
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--reg-alpha", type=float, default=0.0)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    
    #choose if the GBReweighter should be used to reweight background (pT, eta) to match signal
    #default is true
    p.add_argument("--use-gbreweighter", type=str, default="true", choices=["true", "false"])

    #reweighting hyperparameters
    p.add_argument("--gbrw-n-estimators", type=int, default=80)
    p.add_argument("--gbrw-max-depth", type=int, default=3)
    p.add_argument("--gbrw-learning-rate", type=float, default=0.1)
    p.add_argument("--gbrw-min-samples-leaf", type=int, default=200)
    p.add_argument("--gbrw-clip", type=float, default=100.0)
    return p.parse_args()

#function to generate the list of feature column names
def feature_names(K, use_neighbors, use_muon_d0, use_nbr_d0, use_isolation, use_zeta):
    names = ["pt", "eta", "abs_z0sinTheta"]
    if use_muon_d0:
        names.append("d0")
    if use_isolation:
        names.append("isolation")
    if use_neighbors:
        nbh_vars = (["d0"] if use_nbr_d0 else []) + ["pt", "deta", "deltaR", "dz0sintheta"]
        orderings = ("dR", "pT") + (("zeta",) if use_zeta else ())
        #neighbor slot names follow <var>_nbh_<ordering>_<slot>, e.g. pt_nbh_dR_03
        for ordering in orderings:
            for slot in range(1, K + 1):
                names.extend(f"{v}_nbh_{ordering}_{slot:02d}" for v in nbh_vars)
    return names


#function to build the feature matrix, one row per candidate muon (label 1=signal, 0=background)
def build_muon_matrix(df, min_pt, dr_cut, dz_cut, K, 
                      use_neighbors, use_muon_d0, use_nbr_d0, use_isolation, use_zeta,
                      require_single_muon, label):
    
    #column book-keeping for the feature matrix layout (must match feature_names)
    n_per_slot = 5 if use_nbr_d0 else 4
    n_base = 3 + (1 if use_muon_d0 else 0) + (1 if use_isolation else 0)
    n_orderings = 2 + (1 if use_zeta else 0)
    n_cols = n_base + (n_orderings * K * n_per_slot if use_neighbors else 0)
    need_cone = use_neighbors or use_isolation

    #lists to hold the feature rows, labels, and event ids for all events
    #will be stacked at the end
    Xc, yc, ec = [], [], []

    #loop over events; within an event all muon-track pairs are handled with vectorized operations
    #here eid is event number and ev is dataframe of event tracks
    for eid, ev in tqdm(df.groupby(level="entry"), desc=f"build label={label}"):

        #extracting values from the event dataframe into numpy arrays for vectorized operations
        eta = ev["InDetTrack_eta"].values
        phi = ev["InDetTrack_phi"].values
        pt = ev["InDetTrack_pt"].values
        z0s = ev["InDetTrack_z0sinTheta"].values
        z0a = np.abs(z0s)
        d0 = ev["InDetTrack_d0"].values
        ismu = ev["isMuon"].values == True

        #skips signal events without exactly one muon
        #candidates are muons above the pT floor
        if require_single_muon and ismu.sum() != 1:
            continue #all background events are kept, even if they have multiple muons
        usable = pt >= min_pt #qualifies muon candidates and tracks for cone membership

        #cand is integer array of track indices that are muons and pass the pT floor
        #stands for "candidate"
        cand = np.where(ismu & usable)[0] 
        
        #skips empty events (no muons above the pT floor)
        if cand.size == 0:
            continue

        #extract muon-level features for the candidate muons
        #these are the first columns of the feature matrix
        eta_mu, phi_mu, z0s_mu, pt_mu = eta[cand], phi[cand], z0s[cand], pt[cand]

        #cone mask (muons x tracks): deltaR cut and dz cut relative to the muon (signed z0sinTheta)
        #sub object is the cone subset itself
        #set first to none in case neighbor features and isolation are not used (when no cone is needed)
        sub = dRc = None
        if need_cone:
            dRc = compute_deltaR_rect(eta_mu, phi_mu, eta, phi)
            dz = np.abs(z0s_mu[:, None] - z0s[None, :])

            #assemble mask for tracks that are within the cone and usable
            sub = (dRc < dr_cut) & (dz < dz_cut) & usable[None, :]
            #muon's own track is excluded
            sub[np.arange(cand.size), cand] = False

        #feature rows for this event eid
        #NaN-padded by default (XGBoost handles NaN natively) to handle absences of neighbors in the cone
        chunk = np.full((cand.size, n_cols), np.nan, dtype=np.float32)

        c = 0
        #all rows, column c
        chunk[:, c] = pt_mu;       c += 1
        chunk[:, c] = eta_mu;      c += 1
        #right now muon's transverse impact parameter is magntitude; should change this to signed
        chunk[:, c] = z0a[cand];   c += 1
        if use_muon_d0:
            chunk[:, c] = d0[cand]; c += 1
        #relative track isolation = sum of cone track pT / muon pT
        if use_isolation:
            chunk[:, c] = (pt[None, :] * sub).sum(axis=1) / pt_mu; c += 1

        #neighbor slots: up to K cone tracks per ordering (dR ascending, pT descending, optional zeta)
        if use_neighbors and K > 0:
            #for each ordering, sort the cone tracks and take the first K
            #fill the neighbor slots with these tracks
            #dr ascending ordering
            order_dr = np.argsort(np.where(sub, dRc, np.inf), axis=1, kind="stable")[:, :K]
            valid_dr = np.take_along_axis(sub, order_dr, axis=1)

            #pT descending ordering
            pt_row = np.broadcast_to(pt[None, :], sub.shape)
            order_pt = np.argsort(-np.where(sub, pt_row, -np.inf), axis=1, kind="stable")[:, :K]
            valid_pt = np.take_along_axis(sub, order_pt, axis=1)

            #ordering groups of starting column, ranking, validity
            groups = [(n_base, order_dr, valid_dr),
                      (n_base + K * n_per_slot, order_pt, valid_pt)]
            #optional zeta measure ordering lives here
            if use_zeta:
                zeta = np.sqrt((dRc * 20.0) ** 2 + dz ** 2)
                order_z = np.argsort(np.where(sub, zeta, np.inf), axis=1, kind="stable")[:, :K]
                valid_z = np.take_along_axis(sub, order_z, axis=1)
                groups.append((n_base + 2 * K * n_per_slot, order_z, valid_z))

            #function to gather track values into neighbor slots; empty slots set to NaN
            def gather(vals_1d, order, valid):
                g = vals_1d[order].astype(np.float32)
                g[~valid] = np.nan
                return g

            #fill neighbor columns for each ordering; deta and dz0sintheta are relative to the muon
            for base, order, valid in groups:
                dr_g = np.take_along_axis(dRc, order, axis=1).astype(np.float32); dr_g[~valid] = np.nan
                end = base + order.shape[1] * n_per_slot
                fo = 0
                if use_nbr_d0:
                    chunk[:, base + fo:end:n_per_slot] = gather(d0, order, valid); fo += 1
                chunk[:, base + fo:end:n_per_slot] = gather(pt, order, valid);  fo += 1
                chunk[:, base + fo:end:n_per_slot] = gather(eta, order, valid) - eta_mu[:, None]; fo += 1
                chunk[:, base + fo:end:n_per_slot] = dr_g;                      fo += 1
                chunk[:, base + fo:end:n_per_slot] = gather(z0s, order, valid) - z0s_mu[:, None]

        Xc.append(chunk)
        yc.append(np.full(cand.size, label, dtype=np.int64))
        ec.append(np.full(cand.size, int(eid), dtype=np.int64))

    #stack all events into single arrays: features X, labels y, event ids
    if not Xc:
        return np.empty((0, n_cols), np.float32), np.empty(0, np.int64), np.empty(0, np.int64)
    return np.vstack(Xc), np.concatenate(yc), np.concatenate(ec)


#function to load track branches from a sample's root file into a dataframe
#n_events <= 0 loads all
def load_sample(path, n_events):
    f = uproot.open(path + "OutputIsolation.root:OutputIsolation")
    stop = None if n_events <= 0 else n_events
    a = f.arrays(["InDetTrack_eta", "InDetTrack_phi", "InDetTrack_pt",
                  "InDetTrack_z0sinTheta", "InDetTrack_d0", "isMuon"], entry_stop=stop)
    return ak.to_dataframe(a)


#function to split muons into train/holdout by event id
#done so muons from the same event never cross the split
def split_by_event(X, y, eids, holdout_size, seed):
    ev = np.unique(eids)
    tr_ev, ho_ev = train_test_split(ev, test_size=holdout_size, random_state=seed)
    return (X[np.isin(eids, tr_ev)], y[np.isin(eids, tr_ev)]), \
           (X[np.isin(eids, ho_ev)], y[np.isin(eids, ho_ev)])


#function to compute per-muon weights making background (pT, eta) match signal
#fit on training data only
def compute_gb_weights(pt, eta, y, n_estimators, max_depth, learning_rate, min_samples_leaf, clip):
    sig, bkg = y == 1, y == 0
    sig_feats = np.column_stack([pt[sig], eta[sig]])
    bkg_feats = np.column_stack([pt[bkg], eta[bkg]])
    print(f"  fitting GBReweighter: n_sig={sig.sum()}, n_bkg={bkg.sum()}")
    rw = GBReweighter(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                      min_samples_leaf=min_samples_leaf, gb_args={"subsample": 0.6})
    rw.fit(original=bkg_feats, target=sig_feats)

    #clip extreme weights so no single event dominates
    #normalize background weights to mean 1
    bkg_w = np.clip(rw.predict_weights(bkg_feats), 1.0 / clip, clip)
    w = np.ones_like(y, dtype=float)
    w[bkg] = bkg_w
    w[bkg] /= w[bkg].mean()
    return w

#function to compute the Kolmogorov-Smirnov statistic between two weighted samples (after-reweighting check)
#using this for generated csv file to check if reweighting worked
def weighted_ks(x1, w1, x2, w2):
    xs = np.concatenate([x1, x2])
    ws1 = np.concatenate([w1, np.zeros_like(x2)])
    ws2 = np.concatenate([np.zeros_like(x1), w2])
    order = np.argsort(xs)
    c1 = np.cumsum(ws1[order]) / ws1.sum()
    c2 = np.cumsum(ws2[order]) / ws2.sum()
    return float(np.max(np.abs(c1 - c2)))

#function to sum feature importances over the K slots of each neighbor (variable, ordering) group
#otherwise there would over a hundred feature importances
def aggregate_neighbor_importances(feature_names, importances):
    base_pairs, nbr_buckets, per_slot_rows = [], {}, []
    for name, imp in zip(feature_names, importances):
        if "_nbh_" in name:
            gk = name.rsplit("_", 1)[0]
            nbr_buckets[gk] = nbr_buckets.get(gk, 0.0) + float(imp)
            per_slot_rows.append((name, gk, float(imp)))
        else:
            base_pairs.append((name, float(imp)))
            per_slot_rows.append((name, None, float(imp)))
    return base_pairs + sorted(nbr_buckets.items(), key=lambda kv: -kv[1]), per_slot_rows

#function to plot train and holdout roc curves
#uses Youden J operating points; returns both auc values
def plot_roc(y_train, proba_train, y_hold, proba_hold, plot_dir, settings_str):
    fpr_tr, tpr_tr, th_tr = roc_curve(y_train, proba_train)
    auc_tr = float(auc(fpr_tr, tpr_tr)); k_tr = int(np.argmax(tpr_tr - fpr_tr))
    fpr_ho, tpr_ho, th_ho = roc_curve(y_hold, proba_hold)
    auc_ho = float(auc(fpr_ho, tpr_ho)); k_ho = int(np.argmax(tpr_ho - fpr_ho))
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
    ax.set_title("Prompt vs Non-Prompt Muon Classifier — ROC", fontsize=12)
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); add_settings_box(fig, settings_str)
    out = os.path.join(plot_dir, "roc.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(); print(f"saved {out}")
    return auc_tr, auc_ho, {"train_thr": float(th_tr[k_tr]), "hold_thr": float(th_ho[k_ho]),
                            "train_J": float(tpr_tr[k_tr] - fpr_tr[k_tr]), "hold_J": float(tpr_ho[k_ho] - fpr_ho[k_ho])}


#function to plot feature importances (neighbor groups summed)
#also writes per-slot csv
def plot_feature_importances(model, names, n_events, label, plot_dir, settings_str):
    group_pairs, per_slot_rows = aggregate_neighbor_importances(names, model.feature_importances_)
    group_pairs.sort(key=lambda kv: -kv[1])
    disp = [display_name(p[0]) for p in group_pairs]
    imps = np.array([p[1] for p in group_pairs])
    fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(disp))))
    ax.barh(range(len(disp)), imps, color="steelblue", edgecolor="black")
    ax.set_yticks(range(len(disp))); ax.set_yticklabels(disp, fontsize=10)
    ax.set_xlabel("Feature Importance (gain, neighbor groups summed)", fontsize=12)
    ax.set_title(f"Feature Importances — Prompt vs Non-Prompt Muon BDT\n{label}", fontsize=11)
    ax.invert_yaxis(); ax.grid(alpha=0.3, axis="x"); plt.tight_layout()
    nbh_rows = [(n, g) for n, g, _ in per_slot_rows if g is not None]
    key_text = None
    if nbh_rows:
        K = max(int(n.rsplit("_", 1)[1]) for n, _ in nbh_rows)
        orderings = {g.split("_nbh_")[1] for _, g in nbh_rows}     
        lines = [f"neighbor feature key (nbh = up to K={K} tracks in the neighbor cone, same event)"]
        if "dR" in orderings:
            lines.append("  <var>_nbh_dR   : ranked by ΔR ascending (closest first)")
        if "pT" in orderings:
            lines.append("  <var>_nbh_pT   : ranked by pT descending (hardest first)")
        if "zeta" in orderings:
            lines.append("  <var>_nbh_zeta : ranked by ζ=sqrt((20*ΔR)^2+Δz0sinθ^2) ascending (nearest first)")
        lines.append("  each nbh bar = sum of gain over all K slots of that (var, ordering)")
        key_text = "\n".join(lines)
    add_settings_box(fig, settings_str, extra_text=key_text)
    out = os.path.join(plot_dir, "importances.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(); print(f"saved {out}")
    with open(os.path.join(plot_dir, "feature_importances_full.csv"), "w") as fh:
        cw = csv.writer(fh); cw.writerow(["feature_name", "display_name", "neighbor_group", "importance"])
        for name, group, imp in per_slot_rows:
            cw.writerow([name, display_name(name), group or "", f"{imp:.6f}"])


#function to plot the holdout bdt score distributions for signal and background
def plot_score_distribution(y_test, proba, n_events, label, plot_dir, settings_str):
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.linspace(0, 1, 60)
    ax.hist(proba[y_test == 1], bins=bins, alpha=0.6, color="blue", label=SIG_LABEL,
            density=True, histtype="stepfilled")
    ax.hist(proba[y_test == 0], bins=bins, alpha=0.6, color="gold", label=BKG_LABEL,
            density=True, histtype="stepfilled")
    ax.set_xlabel("BDT Score (prompt probability)", fontsize=12); ax.set_ylabel("Normalized Counts", fontsize=12)
    ax.set_title(f"Holdout BDT Score — Prompt vs Non-Prompt Muons\n{label}", fontsize=11)
    ax.legend(fontsize=11); ax.grid(alpha=0.3); plt.tight_layout(); add_settings_box(fig, settings_str)
    out = os.path.join(plot_dir, f"score_distribution_Ev{n_events}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(); print(f"saved {out}")


#function to plot muon (pT, eta) distributions before/after reweighting to verify the reweighter
def plot_pt_eta_reweighted(pt_tr, eta_tr, y_tr, w, plot_dir, settings_str):
    pt_g = pt_tr / 1000.0
    sig, bkg = y_tr == 1, y_tr == 0
    pt_lo = max(np.min(pt_g[pt_g > 0]), 0.01); pt_hi = np.quantile(pt_g, 0.995)
    bins_pt = np.logspace(np.log10(pt_lo), np.log10(max(pt_hi, pt_lo * 2)), 60)
    bins_eta = np.linspace(np.min(eta_tr), np.max(eta_tr), 60)
    for vals, bins, xlabel, logx, fname in (
        (pt_g, bins_pt, "pT [GeV]", True, "verify_pt_gbrw.png"),
        (eta_tr, bins_eta, "eta", False, "verify_eta_gbrw.png"),
    ):
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        for j, (title, wt) in enumerate([("Unweighted", None), ("After GBReweighter", w)]):
            ax[j].hist(vals[sig], bins=bins, density=True, alpha=0.6, label=SIG_LABEL, color="blue",
                       weights=None if wt is None else wt[sig])
            ax[j].hist(vals[bkg], bins=bins, density=True, alpha=0.6, label=BKG_LABEL, color="gold",
                       weights=None if wt is None else wt[bkg])
            if logx: ax[j].set_xscale("log")
            ax[j].set_title(f"Muon {xlabel} — {title}"); ax[j].set_xlabel(xlabel); ax[j].set_ylabel("Density")
            ax[j].legend(); ax[j].grid(alpha=0.3)
        fig.suptitle("GBReweighter Check: Background Reweighted to Match Signal in (pT, eta)", fontsize=12)
        plt.tight_layout(); add_settings_box(fig, settings_str)
        out = os.path.join(plot_dir, fname)
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(); print(f"saved {out}")


#function to plot signal/background distributions for every feature; neighbor slots pooled by (variable, ordering) group
def plot_all_feature_distributions(X_train, y_train, w, names, plot_dir, n_events, settings_str):
    sig, bkg = y_train == 1, y_train == 0
    weighted = not np.allclose(w, 1.0)
    #sort feature columns into muon-level features and neighbor groups
    base_cols, nbr_groups = [], {}
    for i, fname in enumerate(names):
        if "_nbh_" in fname:
            nbr_groups.setdefault(fname.rsplit("_", 1)[0], []).append(i)
        else:
            base_cols.append((i, fname))

    #function to choose histogram binning per feature (log-spaced for pT)
    def _bins_for(values, fname=None):
        if values.size == 0:
            return np.linspace(0.0, 1.0, 60), "", False
        x = values; xlabel = display_name(fname) if fname else ""
        if fname == "pt":
            x = values / 1000.0; pos = x[x > 0]
            lo = max(float(np.min(pos)), 0.01) if pos.size else 0.01
            hi = float(np.quantile(x, 0.995))
            return np.logspace(np.log10(lo), np.log10(max(hi, lo * 2)), 60), "pT [GeV]", True
        if fname == "eta":
            return np.linspace(float(np.min(x)), float(np.max(x)), 60), xlabel, False
        lo = float(np.min(x)); hi = float(np.quantile(x, 0.99))
        if hi <= lo: hi = lo + 1.0
        return np.linspace(lo, hi, 60), xlabel, False

    #function to draw one feature's histogram (two panels when reweighting is on)
    def _emit(sig_v, bkg_v, sig_w, bkg_w, bins, xlabel, logx, prefix, out_name):
        if sig_v.size == 0 and bkg_v.size == 0:
            return
        if weighted:
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            for j, (title, sw, bw) in enumerate([("Unweighted", None, None), ("After GBReweighter", sig_w, bkg_w)]):
                ax[j].hist(sig_v, bins=bins, density=True, alpha=0.6, label=SIG_LABEL, color="blue", weights=sw)
                ax[j].hist(bkg_v, bins=bins, density=True, alpha=0.6, label=BKG_LABEL, color="gold", weights=bw)
                if logx: ax[j].set_xscale("log")
                ax[j].set_title(f"{prefix} — {title}"); ax[j].set_xlabel(xlabel or prefix); ax[j].set_ylabel("Density")
                ax[j].legend(); ax[j].grid(alpha=0.3)
            fig.suptitle(prefix, fontsize=12)
        else:
            fig, a = plt.subplots(figsize=(10, 5))
            a.hist(sig_v, bins=bins, density=True, alpha=0.6, label=SIG_LABEL, color="blue")
            a.hist(bkg_v, bins=bins, density=True, alpha=0.6, label=BKG_LABEL, color="gold")
            if logx: a.set_xscale("log")
            a.set_title(prefix); a.set_xlabel(xlabel or prefix); a.set_ylabel("Density")
            a.legend(); a.grid(alpha=0.3)
        add_settings_box(fig, settings_str); plt.tight_layout()
        out = os.path.join(plot_dir, out_name)
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(); print(f"saved {out}")

    #one plot per muon-level feature
    for col, fname in base_cols:
        x = X_train[:, col]; valid = ~np.isnan(x)
        si, bi = valid & sig, valid & bkg
        sv, bv = x[si], x[bi]
        bins, xlabel, logx = _bins_for(np.concatenate([sv, bv]), fname=fname)
        if fname == "pt": sv, bv = sv / 1000.0, bv / 1000.0
        _emit(sv, bv, w[si], w[bi], bins, xlabel, logx, f"Muon {display_name(fname)}", f"feat_{fname}_Ev{n_events}.png")

    #one plot per neighbor (variable, ordering) group, all K slots pooled
    for gk, cols in nbr_groups.items():
        sub = X_train[:, cols]
        sw = np.broadcast_to(w[sig, None], (int(sig.sum()), sub.shape[1])).ravel()
        bw = np.broadcast_to(w[bkg, None], (int(bkg.sum()), sub.shape[1])).ravel()
        sflat, bflat = sub[sig].ravel(), sub[bkg].ravel()
        sk, bk = ~np.isnan(sflat), ~np.isnan(bflat)
        sv, bv = sflat[sk], bflat[bk]
        phys = gk.split("_nbh_")[0]
        bins, xlabel, logx = _bins_for(np.concatenate([sv, bv]), fname=phys)
        if phys == "pt": sv, bv = sv / 1000.0, bv / 1000.0
        _emit(sv, bv, sw[sk], bw[bk], bins, xlabel, logx, nbh_group_title(gk), f"feat_{gk}_Ev{n_events}.png")


#hyperparameters that are integers when parsed from a --config string; the rest are floats
INT_HP_KEYS = {"n_estimators", "max_depth"}


#function to parse a --config key=value,... string into a full hyperparameter dictionary, starting from the command line values
def parse_config(spec, base):
    hp = dict(base)
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(f"--config token '{tok}' must be key=value")
        k, v = (s.strip() for s in tok.split("=", 1))
        if k not in hp:
            raise ValueError(f"--config key '{k}' is not a hyperparameter; allowed: {sorted(hp)}")
        hp[k] = int(v) if k in INT_HP_KEYS else float(v)
    return hp


#function to collect the xgboost hyperparameters from parsed command line arguments into a dictionary
def hp_from_args(args):
    return dict(n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate,
                subsample=args.subsample, colsample_bytree=args.colsample_bytree,
                min_child_weight=args.min_child_weight, gamma=args.gamma,
                reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda)


#function to construct the xgboost classifier from hyperparameter dictionary
def make_clf(hp, random_state):
    return xgb.XGBClassifier(
        n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], learning_rate=hp["learning_rate"],
        subsample=hp["subsample"], colsample_bytree=hp["colsample_bytree"],
        min_child_weight=hp["min_child_weight"], gamma=hp["gamma"],
        reg_alpha=hp["reg_alpha"], reg_lambda=hp["reg_lambda"],
        objective="binary:logistic", eval_metric="auc", tree_method="hist",
        random_state=random_state, n_jobs=1)


#main function
#load and build both samples, split by event, reweight, then fit and evaluate configs
def main():
    args = parse_args()

    #convert string flags to booleans; d0-mode sets the muon and neighbor d0 switches
    use_neighbors = args.use_neighbors == "true"
    use_gbrw = args.use_gbreweighter == "true"
    d0_mode = args.d0_mode
    use_muon_d0 = d0_mode == "both"
    use_nbr_d0 = d0_mode in ("both", "muon-off")
    use_isolation = args.use_isolation == "true"
    use_zeta = args.use_zeta_order == "true"
    K = args.max_neighbors
    names = feature_names(K, use_neighbors, use_muon_d0, use_nbr_d0, use_isolation, use_zeta)
    ev_tag = "MAX" if args.n_events <= 0 else str(args.n_events)

    #load and build the feature matrix for each sample; dataframe freed before loading the next
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

    #muon (pT, eta) columns saved for reweighting and verification plots
    pt_train = X_train[:, names.index("pt")].copy()
    eta_train = X_train[:, names.index("eta")].copy()

    #background (pT, eta) reweighting with before/after Kolmogorov-Smirnov check
    w = np.ones_like(y_train, dtype=float)
    ks = None
    if use_gbrw:
        print("\nfitting GBReweighter on (pT, eta)...")
        w = compute_gb_weights(pt_train, eta_train, y_train, args.gbrw_n_estimators,
                               args.gbrw_max_depth, args.gbrw_learning_rate,
                               args.gbrw_min_samples_leaf, args.gbrw_clip)
        sig, bkg = y_train == 1, y_train == 0
        ks = (float(ks_2samp(pt_train[sig], pt_train[bkg]).statistic),
              float(ks_2samp(eta_train[sig], eta_train[bkg]).statistic),
              weighted_ks(pt_train[sig], w[sig], pt_train[bkg], w[bkg]),
              weighted_ks(eta_train[sig], w[sig], eta_train[bkg], w[bkg]))
        print(f"KS pT  unweighted={ks[0]:.4f} reweighted={ks[2]:.4f}")
        print(f"KS eta unweighted={ks[1]:.4f} reweighted={ks[3]:.4f}")

    #class imbalance: upweight signal by n_bkg/n_sig
    spw = n_bkg_tr / max(n_sig_tr, 1)
    final_w = w.copy(); final_w[y_train == 1] *= spw
    print(f"scale_pos_weight (n_bkg/n_sig): {spw:.2f}")

    #run settings text for the plot settings boxes; _row pads labels so the box lines up as a table
    def _row(lbl, val): return f"{lbl:<14}: {val}"
    settings_data = "\n".join([
        f"RUN SETTINGS   (muonBDT.py v{VERSION})",
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
                           f"|Δz0sinθ|<{args.neighbor_dz_cut}mm (signed vertex sep); same cone train/holdout; "
                           f"dR-asc, pT-desc" + (", ζ-asc" if use_zeta else "") + "; NaN-padded")
                          if use_neighbors else "off"),
        _row("Reweighting", "GBReweighter (pT,eta) bkg->signal, train only" if use_gbrw else "off"),
        _row("Imbalance", f"scale_pos_weight={spw:.2f} on signal"),
        _row("Events", f"signal {n_sig_ev:,} events / {len(ys):,} muons, "
                       f"background {n_bkg_ev:,} events / {len(yb):,} muons"),
        _row("Train/holdout", f"{len(y_train):,} / {len(y_hold):,} muons (80/20 split by event)"),
    ])

    #output directory name encodes the cone, feature, and reweighting settings
    #project root is three levels up from this file (muonisolation/main/muon_iso_BDT/)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output")
    nbr_str = f"dr{args.neighbor_dr_cut}_dz{args.neighbor_dz_cut}_K{K}" if use_neighbors else "noNbr"
    nbr_str += {"both": "_d0on", "muon-off": "_d0muOFF", "none": "_d0off"}[d0_mode]
    nbr_str += "_iso" if use_isolation else "_noiso"
    nbr_str += "_zeta" if use_zeta else ""
    nbr_str += "_rw" if use_gbrw else "_norw"

    #function to fit one hyperparameter config, make plots, and write model.pkl and summary.csv
    def fit_eval(hp, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        clf = make_clf(hp, args.random_state)
        clf.fit(X_train, y_train, sample_weight=final_w)
        proba = clf.predict_proba(X_hold)[:, 1]
        proba_train = clf.predict_proba(X_train)[:, 1]
        y_pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_hold, y_pred)
        settings_full = settings_data + "\n" + _row(
            "Hyperparams", f"trees={hp['n_estimators']}, depth={hp['max_depth']}, lr={hp['learning_rate']}, "
                           f"subsample={hp['subsample']}, colsample={hp['colsample_bytree']}, "
                           f"min_child_weight={hp['min_child_weight']}, reg_lambda={hp['reg_lambda']}")
        auc_tr, auc_ho, yj = plot_roc(y_train, proba_train, y_hold, proba, out_dir, settings_full)
        label = f"AUC holdout = {auc_ho:.4f}  (train {auc_tr:.4f})"
        plot_score_distribution(y_hold, proba, ev_tag, label, out_dir, settings_full)
        plot_feature_importances(clf, names, ev_tag, label, out_dir, settings_full)
        with open(os.path.join(out_dir, "model.pkl"), "wb") as fh:
            pickle.dump(clf, fh)
            
        #write all settings and results to summary.csv (used by summarize_runs.py additional plotting script)
        with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
            wcsv = csv.writer(fh); wcsv.writerow(["key", "value"])
            rows = [("muonbdt_version", VERSION),
                    ("n_events_requested", args.n_events), ("min_pt_MeV", args.min_pt),
                    ("neighbor_dr_cut", args.neighbor_dr_cut), ("neighbor_dz_cut_mm", args.neighbor_dz_cut),
                    ("max_neighbors_K", K), ("use_neighbors", use_neighbors),
                    ("use_zeta_order", use_zeta),
                    ("d0_mode", d0_mode), ("use_muon_d0", use_muon_d0), ("use_nbr_d0", use_nbr_d0),
                    ("use_isolation", use_isolation), ("use_gbreweighter", use_gbrw),
                    ("n_features", len(names)),
                    ("n_signal_events", n_sig_ev), ("n_bkg_events", n_bkg_ev),
                    ("n_signal_muons", len(ys)), ("n_bkg_muons", len(yb)),
                    ("n_train", len(y_train)), ("n_holdout", len(y_hold)), ("scale_pos_weight", spw),
                    ("n_estimators", hp["n_estimators"]), ("max_depth", hp["max_depth"]),
                    ("learning_rate", hp["learning_rate"]), ("subsample", hp["subsample"]),
                    ("colsample_bytree", hp["colsample_bytree"]), ("min_child_weight", hp["min_child_weight"]),
                    ("gamma", hp["gamma"]), ("reg_alpha", hp["reg_alpha"]), ("reg_lambda", hp["reg_lambda"]),
                    ("auc_train", auc_tr), ("auc_holdout", auc_ho),
                    ("holdout_youden_thr", yj["hold_thr"]), ("holdout_youden_J", yj["hold_J"]),
                    ("holdout_accuracy_0p5", acc)]
            if ks is not None:
                rows += [("ks_pt_unweighted", ks[0]), ("ks_eta_unweighted", ks[1]),
                         ("ks_pt_reweighted", ks[2]), ("ks_eta_reweighted", ks[3])]
            for k, v in rows:
                wcsv.writerow([k, v])
        print(f"  AUC train {auc_tr:.4f} | holdout {auc_ho:.4f} | gap {auc_tr-auc_ho:.4f}")
        return {"auc_train": auc_tr, "auc_holdout": auc_ho}

    #scan mode if --config was given (features built once, each config fit in sequence); otherwise a single fit
    base_hp = hp_from_args(args)
    configs = [parse_config(c, base_hp) for c in args.config] if args.config else None

    if configs:
        root = os.path.join(output_dir, f"muonbdt_scan_{nbr_str}_v{VERSION}_n{ev_tag}")
        #suffix _runN avoids overwriting an existing output directory
        r = root; i = 0
        while os.path.exists(r):
            i += 1; r = f"{root}_run{i}"
        os.makedirs(r)
        print(f"\nscan root: {r}  ({len(configs)} configs)")
        #data-only distribution plots are the same for every config, generated once into the scan root
        plot_all_feature_distributions(X_train, y_train, w, names, r, ev_tag, settings_data)
        if use_gbrw:
            plot_pt_eta_reweighted(pt_train, eta_train, y_train, w, r, settings_data)
        scan_rows = []
        for idx, hp in enumerate(configs):
            tag = (f"cfg{idx:02d}_d{hp['max_depth']}_n{hp['n_estimators']}_lr{hp['learning_rate']}"
                   f"_L{hp['reg_lambda']}_mcw{hp['min_child_weight']}")
            print(f"\n===== scan config {idx+1}/{len(configs)}: {tag} =====")
            res = fit_eval(hp, os.path.join(r, tag))
            scan_rows.append((tag, hp, res))
            #rewrite the aggregate scan summary after every config so finished ones are visible immediately
            with open(os.path.join(r, "scan_summary.csv"), "w", newline="") as fh:
                cw = csv.writer(fh)
                cw.writerow(["config", "max_depth", "n_estimators", "learning_rate", "reg_lambda",
                             "min_child_weight", "auc_train", "auc_holdout", "train_holdout_gap"])
                for t, h, rr in scan_rows:
                    cw.writerow([t, h["max_depth"], h["n_estimators"], h["learning_rate"], h["reg_lambda"],
                                 h["min_child_weight"], f"{rr['auc_train']:.4f}", f"{rr['auc_holdout']:.4f}",
                                 f"{rr['auc_train']-rr['auc_holdout']:.4f}"])
            print(f"===== config {idx+1}/{len(configs)} DONE -> "
                  f"holdout AUC {res['auc_holdout']:.4f} (train {res['auc_train']:.4f}) =====")
        print("\nall scan configs done")
    else:
        base = os.path.join(output_dir, f"muonbdt_{nbr_str}_v{VERSION}_n{ev_tag}")
        #suffix _runN avoids overwriting an existing output directory
        out = base; i = 0
        while os.path.exists(out):
            i += 1; out = f"{base}_run{i}"
        os.makedirs(out)
        print(f"\noutput dir: {out}")
        plot_all_feature_distributions(X_train, y_train, w, names, out, ev_tag, settings_data)
        if use_gbrw:
            plot_pt_eta_reweighted(pt_train, eta_train, y_train, w, out, settings_data)
        fit_eval(base_hp, out)

    print("done")


#run main() only when executed as a script, not when imported
if __name__ == "__main__":
    main()
