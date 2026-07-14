#!/usr/bin/env python3

import argparse
import os
import csv
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc

SIG_DEFAULT = "/lstr/sahara/niueftracking/kesedlac/muon_isolation/OUTPUT/run_InvPtPU200/"
BKG_DEFAULT = "/lstr/sahara/niueftracking/kesedlac/muon_isolation/OUTPUT/run_bjet/"
SIG_LABEL = "Prompt Muon (Signal)"
BKG_LABEL = "Non-Prompt Muon (Background)"

VERSION = "1.0"


def get_auto_dz_cut_vectorized(pts_gev, etas):
    abs_etas = np.abs(etas)
    eta_bins = np.zeros_like(abs_etas, dtype=int)
    eta_bins[abs_etas >= 1] = 1
    eta_bins[abs_etas >= 2] = 2
    cuts_table = np.array([
        [0.6, 0.6, 0.5], [0.6, 0.6, 0.6], [1.0, 1.0, 0.6], [1.2, 1.2, 1.0],
        [2.0, 1.6, 1.0], [3.3, 2.5, 1.6], [4.2, 3.3, 2.5], [5.4, 5.2, 4.2],
        [6.0, 5.2, 4.2], [6.0, 5.5, 4.2], [6.0, 6.6, 5.4], [6.0, 8.5, 8.5],
    ])
    pt_bins = np.zeros_like(pts_gev, dtype=int)
    for i, edge in enumerate([1.3, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0, 20.0, 30.0], start=1):
        pt_bins[pts_gev >= edge] = i
    return cuts_table[pt_bins, eta_bins]


def compute_deltaR_rect(eta_mu, phi_mu, eta, phi):
    d_eta = eta_mu[:, None] - eta[None, :]
    d_phi = np.arctan2(np.sin(phi_mu[:, None] - phi[None, :]),
                       np.cos(phi_mu[:, None] - phi[None, :]))
    return np.sqrt(d_eta ** 2 + d_phi ** 2)


def parse_args():
    p = argparse.ArgumentParser(description="isolation-cut baseline for prompt vs non-prompt muons")
    p.add_argument("--signal-path", type=str, default=SIG_DEFAULT)
    p.add_argument("--bkg-path", type=str, default=BKG_DEFAULT)
    p.add_argument("--n-events", type=int, default=2000, help="events per sample; <=0 loads all")
    p.add_argument("--min-pt", type=float, default=1000.0, help="pT floor (MeV) for muon and cone tracks")
    p.add_argument("--iso-dr-cut", type=float, default=0.3, help="isolation cone radius in dR")
    p.add_argument("--iso-dz-cut", type=str, default="auto",
                   help="cone dz cut: 'none', 'auto' (pT,|eta| table), or a fixed value in mm")
    return p.parse_args()


def load_sample(path, n_events):
    f = uproot.open(path + "OutputIsolation.root:OutputIsolation")
    stop = None if n_events <= 0 else n_events
    a = f.arrays(["InDetTrack_eta", "InDetTrack_phi", "InDetTrack_pt",
                  "InDetTrack_z0sinTheta", "isMuon"], entry_stop=stop)
    return ak.to_dataframe(a)


def muon_isolations(df, min_pt, dr_cut, dz_mode, dz_val, require_single_muon):
    out = []
    n_ev = 0
    for eid, ev in tqdm(df.groupby(level="entry"), desc="isolation"):
        eta = ev["InDetTrack_eta"].values
        phi = ev["InDetTrack_phi"].values
        pt = ev["InDetTrack_pt"].values
        z0s = ev["InDetTrack_z0sinTheta"].values
        ismu = ev["isMuon"].values == True
        if require_single_muon and ismu.sum() != 1:
            continue
        usable = pt >= min_pt
        cand = np.where(ismu & usable)[0]
        if cand.size == 0:
            continue
        eta_mu, phi_mu, pt_mu = eta[cand], phi[cand], pt[cand]
        dR = compute_deltaR_rect(eta_mu, phi_mu, eta, phi)
        in_cone = (dR > 0) & (dR < dr_cut) & usable[None, :]
        if dz_mode != "none":
            dz_thr = (get_auto_dz_cut_vectorized(pt_mu / 1000.0, eta_mu) if dz_mode == "auto"
                      else np.full(cand.size, dz_val))
            in_cone &= np.abs(z0s[None, :] - z0s[cand][:, None]) < dz_thr[:, None]
        cone_pt = (pt[None, :] * in_cone).sum(axis=1)
        out.append(cone_pt / pt_mu)
        n_ev += 1
    return (np.concatenate(out) if out else np.empty(0)), n_ev


def main():
    args = parse_args()
    ev_tag = "MAX" if args.n_events <= 0 else str(args.n_events)

    dzs = args.iso_dz_cut.strip().lower()
    if dzs in ("none", "off"):
        dz_mode, dz_val, dz_tag = "none", 0.0, "none"
    elif dzs == "auto":
        dz_mode, dz_val, dz_tag = "auto", 0.0, "auto"
    else:
        dz_val = float(args.iso_dz_cut); dz_mode, dz_tag = "fixed", f"{dz_val:g}mm"

    print("loading + isolation: signal (run_InvPtPU200)...")
    df = load_sample(args.signal_path, args.n_events)
    iso_sig, n_sig_ev = muon_isolations(df, args.min_pt, args.iso_dr_cut, dz_mode, dz_val,
                                        require_single_muon=True)
    del df
    print("loading + isolation: background (run_bjet)...")
    df = load_sample(args.bkg_path, args.n_events)
    iso_bkg, n_bkg_ev = muon_isolations(df, args.min_pt, args.iso_dr_cut, dz_mode, dz_val,
                                        require_single_muon=False)
    del df
    print(f"signal muons: {len(iso_sig):,} ({n_sig_ev:,} events) | "
          f"background muons: {len(iso_bkg):,} ({n_bkg_ev:,} events)")

    iso = np.concatenate([iso_sig, iso_bkg])
    y = np.concatenate([np.ones(len(iso_sig), int), np.zeros(len(iso_bkg), int)])
    score = -iso
    auc_iso = float(roc_auc_score(y, score))
    print(f"\nisolation-cut AUC: {auc_iso:.4f}")

    if dz_mode == "auto":
        cone = f"dR < {args.iso_dr_cut} and |z0sinθ_track - z0sinθ_muon| < auto(pT,|eta|) table"
    elif dz_mode == "fixed":
        cone = f"dR < {args.iso_dr_cut} and |z0sinθ_track - z0sinθ_muon| < {dz_val:g} mm"
    else:
        cone = f"dR < {args.iso_dr_cut} (no dz cut)"
    settings = "\n".join([
        f"RUN SETTINGS   (isoPLOT.py v{VERSION})",
        f"{'Samples':<14}: signal=run_InvPtPU200 (prompt), bkg=run_bjet (non-prompt); label=provenance",
        f"{'Variable':<14}: relative track isolation = (sum pT of cone tracks) / pT(muon)",
        f"{'Cone':<14}: {cone}; tracks pT >= {args.min_pt/1000:.1f} GeV, muon excluded",
        f"{'Muons':<14}: signal {len(iso_sig):,} ({n_sig_ev:,} events), background {len(iso_bkg):,} ({n_bkg_ev:,} events)",
    ])

    #project root is three levels up from this file (muonisolation/main/muon_iso_cut/)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "output", f"isoplot_dr{args.iso_dr_cut}_dz{dz_tag}_v{VERSION}_n{ev_tag}")
    o = out; i = 0
    while os.path.exists(o):
        i += 1; o = f"{out}_run{i}"
    os.makedirs(o)
    print(f"output dir: {o}")

    def settings_box(fig):
        fig.text(0.01, -0.03, settings, ha="left", va="top", fontsize=8, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="whitesmoke", edgecolor="gray"))

    fpr, tpr, thr = roc_curve(y, score)
    a = float(auc(fpr, tpr))
    k = int(np.argmax(tpr - fpr))
    iso_thr = -thr[k]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"Isolation Cut ROC (AUC = {a:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random")
    ax.scatter([fpr[k]], [tpr[k]], color="darkorange", s=160, marker="*", edgecolors="black", zorder=5,
               label=f"Youden J (iso < {iso_thr:.3f}, J = {tpr[k]-fpr[k]:.3f})")
    ax.set_xlabel("False Positive Rate (non-prompt muons misidentified as prompt)", fontsize=11)
    ax.set_ylabel("True Positive Rate (prompt muons correctly identified)", fontsize=11)
    ax.set_title("Prompt vs Non-Prompt Muon — Isolation-Cut ROC", fontsize=12)
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); settings_box(fig)
    plt.savefig(os.path.join(o, "roc.png"), dpi=150, bbox_inches="tight"); plt.close()

    hi = float(np.quantile(iso, 0.99))
    bins = np.linspace(0, max(hi, 0.1), 60)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(np.clip(iso_sig, 0, bins[-1]), bins=bins, density=True, alpha=0.6,
            color="blue", label=SIG_LABEL)
    ax.hist(np.clip(iso_bkg, 0, bins[-1]), bins=bins, density=True, alpha=0.6,
            color="gold", label=BKG_LABEL)
    ax.set_xlabel("Relative Track Isolation  (Σ pT in cone / pT muon)", fontsize=12)
    ax.set_ylabel("Normalized Counts", fontsize=12)
    ax.set_title(f"Muon Isolation — Prompt vs Non-Prompt\nisolation-cut AUC = {auc_iso:.4f}", fontsize=11)
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    plt.tight_layout(); settings_box(fig)
    plt.savefig(os.path.join(o, "isolation_distribution.png"), dpi=150, bbox_inches="tight"); plt.close()

    with open(os.path.join(o, "summary.csv"), "w", newline="") as fh:
        cw = csv.writer(fh); cw.writerow(["key", "value"])
        for k2, v2 in [("isoplot_version", VERSION),
                       ("iso_dr_cut", args.iso_dr_cut), ("iso_dz_cut_mm", args.iso_dz_cut),
                       ("min_pt_MeV", args.min_pt), ("n_signal_muons", len(iso_sig)),
                       ("n_bkg_muons", len(iso_bkg)), ("n_signal_events", n_sig_ev),
                       ("n_bkg_events", n_bkg_ev), ("auc_isolation_cut", auc_iso),
                       ("youden_iso_cut", iso_thr)]:
            cw.writerow([k2, v2])
    print("saved roc.png, isolation_distribution.png, summary.csv\ndone")


if __name__ == "__main__":
    main()
