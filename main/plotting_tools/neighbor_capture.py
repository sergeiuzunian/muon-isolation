#!/usr/bin/env python3

#neighbor_capture.py - Captured Neighbor Slots vs Cone dR
#counts cone tracks per muon for a grid of dR cuts and K values and writes a csv; summarize_runs.py plots it

#modules for command line argument parsing, file handling, and numerical operations
import argparse
import os
import sys
import csv
import numpy as np

#module for progress bar in terminal output
from tqdm import tqdm

#muonBDT.py is imported from its directory for the sample paths, loader, and deltaR
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "muon_iso_BDT"))
from muonBDT import load_sample, compute_deltaR_rect, SIG_DEFAULT, BKG_DEFAULT

#cone dR grid to scan
DR_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0]


#function to count cone tracks per candidate muon (same dz cut and pT floor as muonBDT.py)
#returns per (dR, K) the mean number of filled slots and the fraction of muons with all K slots filled
def capture_stats(df, drs, min_pt, dz_cut, Ks, require_single_muon):
    Ka = np.asarray(Ks)
    sums = np.zeros((len(drs), len(Ka)))
    sat = np.zeros((len(drs), len(Ka)))
    n_mu = 0
    for eid, ev in tqdm(df.groupby(level="entry"), desc="count"):
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
        dRc = compute_deltaR_rect(eta[cand], phi[cand], eta, phi)
        base = (np.abs(z0s[cand][:, None] - z0s[None, :]) < dz_cut) & usable[None, :]
        base[np.arange(cand.size), cand] = False
        for i, dr in enumerate(drs):
            cnt = ((dRc < dr) & base).sum(axis=1)
            sums[i] += np.minimum(cnt[:, None], Ka[None, :]).sum(axis=0)
            sat[i] += (cnt[:, None] >= Ka[None, :]).sum(axis=0)
        n_mu += cand.size
    return sums / max(n_mu, 1), sat / max(n_mu, 1), n_mu


#main function
#count for both samples and write neighbor_capture.csv
def main():
    p = argparse.ArgumentParser(description="captured neighbor slots vs cone dR, for one or more K")
    p.add_argument("--signal-path", type=str, default=SIG_DEFAULT)
    p.add_argument("--bkg-path", type=str, default=BKG_DEFAULT)
    p.add_argument("--n-events", type=int, default=0, help="events per sample; <=0 loads all")
    p.add_argument("--min-pt", type=float, default=1000.0)
    p.add_argument("--dz-cut", type=float, default=15.0)
    p.add_argument("--max-neighbors", type=int, nargs="+", default=[10, 20, 30, 40, 50],
                   help="one or more K values; all are computed in a single data pass")
    args = p.parse_args()

    #project root is three levels up from this file (muonisolation/main/plotting_tools/)
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "output", "neighbor_capture")
    os.makedirs(outdir, exist_ok=True)
    Ks = sorted(set(args.max_neighbors))

    rows = []
    for name, path, single in [("signal", args.signal_path, True),
                               ("background", args.bkg_path, False)]:
        print(f"loading {name}...")
        df = load_sample(path, args.n_events)
        mean_filled, frac_sat, n_mu = capture_stats(df, DR_GRID, args.min_pt, args.dz_cut, Ks, single)
        del df
        print(f"  {name}: {n_mu:,} muons")
        for i, dr in enumerate(DR_GRID):
            for j, K in enumerate(Ks):
                rows.append([name, dr, K, f"{mean_filled[i, j]:.4f}",
                             f"{frac_sat[i, j]:.4f}", n_mu, args.dz_cut])

    out = os.path.join(outdir, "neighbor_capture.csv")
    with open(out, "w", newline="") as fh:
        cw = csv.writer(fh)
        cw.writerow(["sample", "dr", "K", "mean_filled_slots", "frac_saturated", "n_muons", "dz_cut"])
        cw.writerows(rows)
    print(f"saved {out}\ndone")


#run main() only when executed as a script, not when imported
if __name__ == "__main__":
    main()
