#!/usr/bin/env python3

import argparse
import os
import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIG_DEFAULT = "/lstr/sahara/niueftracking/kesedlac/muon_isolation/OUTPUT/run_InvPtPU200/"
BKG_DEFAULT = "/lstr/sahara/niueftracking/kesedlac/muon_isolation/OUTPUT/run_bjet/"
SIG_LABEL = "Signal — Prompt Muon Sample"
BKG_LABEL = "Background — Non-Prompt Muon Sample"


def load_pt_ismu(path, n_events):
    f = uproot.open(path + "OutputIsolation.root:OutputIsolation")
    stop = None if n_events <= 0 else n_events
    a = f.arrays(["InDetTrack_pt", "isMuon"], entry_stop=stop)
    pt = np.asarray(ak.flatten(a["InDetTrack_pt"]), dtype=float) / 1000.0
    ismu = np.asarray(ak.flatten(a["isMuon"])).astype(bool)
    return pt, ismu


def log_bins(*arrays, n=60, lo_floor=0.1, qhi=0.995):
    pooled = np.concatenate([a[a > 0] for a in arrays])
    lo = max(lo_floor, float(np.min(pooled)))
    hi = float(np.quantile(pooled, qhi))
    return np.logspace(np.log10(lo), np.log10(max(hi, lo * 2)), n)


def _style(ax):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Track pT [GeV]"); ax.set_ylabel("Normalized Counts")
    ax.axvline(1.0, color="gray", ls="--", lw=1, alpha=0.7, label="BDT Track Floor (1 GeV)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")


def main():
    p = argparse.ArgumentParser(description="track pT spectra: signal vs background")
    p.add_argument("--signal-path", type=str, default=SIG_DEFAULT)
    p.add_argument("--bkg-path", type=str, default=BKG_DEFAULT)
    p.add_argument("--n-events", type=int, default=3000, help="events per sample; <=0 loads all")
    args = p.parse_args()

    #project root is three levels up from this file (muonisolation/main/plotting_tools/)
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "output", "sample_spectra")
    os.makedirs(outdir, exist_ok=True)

    print("loading signal (run_InvPtPU200)...")
    pt_s, mu_s = load_pt_ismu(args.signal_path, args.n_events)
    print("loading background (run_bjet)...")
    pt_b, mu_b = load_pt_ismu(args.bkg_path, args.n_events)

    def stats(name, pt, mu):
        nm = pt[~mu]
        print(f"  {name:10s}: {pt.size:,} tracks ({mu.sum():,} muon), "
              f"non-muon median pT = {np.median(nm):.2f} GeV, "
              f"frac non-muon pT>5 GeV = {(nm > 5).mean():.3f}")
    stats("signal", pt_s, mu_s)
    stats("background", pt_b, mu_b)

    bins = log_bins(pt_s, pt_b)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    ax[0].hist(pt_s, bins=bins, density=True, color="steelblue", alpha=0.8,
               label=f"All Tracks (N = {pt_s.size:,})")
    ax[0].set_title(SIG_LABEL)
    ax[1].hist(pt_b, bins=bins, density=True, color="goldenrod", alpha=0.8,
               label=f"All Tracks (N = {pt_b.size:,})")
    ax[1].set_title(BKG_LABEL)
    for a in ax:
        _style(a)
    fig.suptitle("Track pT Spectra: All Tracks", fontsize=14, weight="bold")
    fig.tight_layout()
    f1 = os.path.join(outdir, "pt_spectra_all_tracks.png")
    fig.savefig(f1, dpi=150); plt.close(fig); print(f"saved {f1}")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    for a, pt, mu, lab in [(ax[0], pt_s, mu_s, SIG_LABEL), (ax[1], pt_b, mu_b, BKG_LABEL)]:
        a.hist(pt[~mu], bins=bins, density=True, color="steelblue", alpha=0.6,
               label=f"Non-Muon Tracks (N = {(~mu).sum():,})")
        a.hist(pt[mu], bins=bins, density=True, color="crimson", alpha=0.6,
               label=f"Muon Tracks (N = {mu.sum():,})")
        a.set_title(lab)
        _style(a)
    fig.suptitle("Track pT Spectra: Muon vs Non-Muon Tracks", fontsize=14, weight="bold")
    fig.tight_layout()
    f2 = os.path.join(outdir, "pt_spectra_muon_vs_nonmuon.png")
    fig.savefig(f2, dpi=150); plt.close(fig); print(f"saved {f2}")

    print("done")


if __name__ == "__main__":
    main()
