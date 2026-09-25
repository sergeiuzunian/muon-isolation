#!/usr/bin/env python3

#isoPLOT_vary.py - Isolation-Cut AUC vs Cone dR for |dz0sinTheta| Window Table Variants
#scales the auto dz table of isoPLOT.py in pT, eta, or window size and compares the isolation-cut auc

#modules for file handling, command line argument parsing, and wrapping long lines in the settings box
import os
import sys
import csv
import argparse
import textwrap

#modules for vectorized numerical operations and plotting (Agg backend, no display on the batch nodes)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#module for progress bar in terminal output
from tqdm import tqdm

#roc curve and auc for the isolation cut
from sklearn.metrics import roc_curve, auc

#isoPLOT.py from this directory provides the sample paths, loader, and deltaR
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import isoPLOT

#isoPLOT_vary.py version for keeping track of run data relative to script edits
VERSION = "1.0"

#default |z0sinTheta| window table in mm, same as isoPLOT.py; rows are pT bins, columns are |eta| bins
BASE_TABLE = np.array([
    [0.6, 0.6, 0.5], [0.6, 0.6, 0.6], [1.0, 1.0, 0.6], [1.2, 1.2, 1.0],
    [2.0, 1.6, 1.0], [3.3, 2.5, 1.6], [4.2, 3.3, 2.5], [5.4, 5.2, 4.2],
    [6.0, 5.2, 4.2], [6.0, 5.5, 4.2], [6.0, 6.6, 5.4], [6.0, 8.5, 8.5],
])
BASE_ETA_EDGES = np.array([1.0, 2.0])
BASE_PT_EDGES = np.array([1.3, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0, 20.0, 30.0])

#table variants: scale the eta edges, the pT edges, or the window entries
#track-binned looks the window up per track instead of per muon; MODE_DESC feeds titles and the settings box
MODES = ["default", "eta-half", "eta-double", "pt-half", "pt-3rd", "pt-3.5th",
         "pt-4th", "pt-5th", "pt-6th", "pt-7th", "pt-8th", "pt-double",
         "window-half", "window-double", "track-binned"]

PT_DIVISORS = {"pt-half": 2.0, "pt-3rd": 3.0, "pt-3.5th": 3.5, "pt-4th": 4.0,
               "pt-5th": 5.0, "pt-6th": 6.0, "pt-7th": 7.0, "pt-8th": 8.0}

MODE_DESC = {
    "default_window": "default table (reference)",
    "default": "default table",
    "eta-half": "table eta values halved",
    "eta-double": "table eta values doubled",
    "pt-half": "table pT values halved",
    "pt-3rd": "table pT values /3",
    "pt-3.5th": "table pT values /3.5",
    "pt-4th": "table pT values /4",
    "pt-5th": "table pT values /5",
    "pt-6th": "table pT values /6",
    "pt-7th": "table pT values /7",
    "pt-8th": "table pT values /8",
    "pt-double": "table pT values doubled",
    "window-half": "table window entries halved",
    "window-double": "table window entries doubled",
    "track-binned": "default table binned by track pT and eta",
}


#function to build one table variant (table, edges, per-track flag) from its mode name
def make_variant(mode):
    table = BASE_TABLE.copy()
    eta_edges = BASE_ETA_EDGES.copy()
    pt_edges = BASE_PT_EDGES.copy()
    per_track = mode == "track-binned"
    if mode == "eta-half":
        eta_edges = eta_edges / 2.0
    elif mode == "eta-double":
        eta_edges = eta_edges * 2.0
    elif mode in PT_DIVISORS:
        pt_edges = pt_edges / PT_DIVISORS[mode]
    elif mode == "pt-double":
        pt_edges = pt_edges * 2.0
    elif mode == "window-half":
        table = table / 2.0
    elif mode == "window-double":
        table = table * 2.0
    return {"table": table, "eta_edges": eta_edges, "pt_edges": pt_edges, "per_track": per_track}


#function to look up the window for each (pT, eta) from a variant table
def dz_window(pts_gev, etas, v):
    eta_bins = np.zeros(np.shape(etas), dtype=int)
    for i, e in enumerate(v["eta_edges"], start=1):
        eta_bins[np.abs(etas) >= e] = i
    pt_bins = np.zeros(np.shape(pts_gev), dtype=int)
    for i, e in enumerate(v["pt_edges"], start=1):
        pt_bins[pts_gev >= e] = i
    return v["table"][pt_bins, eta_bins]


#function to compute relative track isolation for every (dR cut, variant) pair in one pass over a sample
#returns a dictionary keyed by (dR, mode) and the number of events used
def isolations(path, n_events, min_pt, dr_cuts, variants, require_single_muon):
    df = isoPLOT.load_sample(path, n_events)
    keys = [(dr, name) for name in variants for dr in dr_cuts]
    out = {k: [] for k in keys}
    n_ev = 0
    for eid, ev in tqdm(df.groupby(level="entry"), desc=os.path.basename(path.rstrip("/"))):
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
        dR = isoPLOT.compute_deltaR_rect(eta_mu, phi_mu, eta, phi)
        adz = np.abs(z0s[None, :] - z0s[cand][:, None])
        base_cone = (dR > 0) & usable[None, :]
        for name, v in variants.items():
            if v["per_track"]:
                thr = dz_window(pt / 1000.0, eta, v)[None, :]
            else:
                thr = dz_window(pt_mu / 1000.0, eta_mu, v)[:, None]
            dz_ok = adz < thr
            for dr in dr_cuts:
                in_cone = base_cone & (dR < dr) & dz_ok
                out[(dr, name)].append((pt[None, :] * in_cone).sum(axis=1) / pt_mu)
        n_ev += 1
    del df
    return {k: (np.concatenate(v) if v else np.empty(0)) for k, v in out.items()}, n_ev


#function to format a table edge for the row and column labels
def fmt_edge(x):
    return f"{x:g}"


#function to draw one variant table as an image
def render_table(name, v, out_dir):
    table, ee, pe = v["table"], v["eta_edges"], v["pt_edges"]
    rows = [f"< {fmt_edge(pe[0])}"]
    for a, b in zip(pe[:-1], pe[1:]):
        rows.append(f"{fmt_edge(a)}-{fmt_edge(b)}")
    rows.append(f"> {fmt_edge(pe[-1])}")
    cols = [f"|η|<{fmt_edge(ee[0])}",
            f"{fmt_edge(ee[0])}<|η|<{fmt_edge(ee[1])}",
            f"|η|>{fmt_edge(ee[1])}"]
    cell = [[f"{x:.1f}" for x in row] for row in table]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.axis("off")
    tbl = ax.table(cellText=cell, rowLabels=rows, colLabels=cols,
                   cellLoc="center", rowLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.32)
    for (r, c), cl in tbl.get_celld().items():
        cl.set_linewidth(0.8)
        edges = ""
        if r == 0:
            edges += "TB"
            cl.set_text_props(weight="bold")
        if r == len(cell):
            edges += "B"
        if c >= 0:
            edges += "L"
        if c == table.shape[1] - 1:
            edges += "R"
        cl.visible_edges = edges
        if c == -1:
            cl.visible_edges = "R" + ("TB" if r == 0 else "") + ("B" if r == len(cell) else "")
    ax.set_title(f"{name} - {MODE_DESC.get(name, name)}\n"
                 f"per-track $|\\Delta z_0\\sin\\theta|$ window [mm], binned in $p_T$ [GeV] and $|\\eta|$",
                 fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


#function to plot the isolation-cut roc for one variant and dR cut; returns the auc
def render_roc(name, dr, s, b, out_dir):
    y = np.concatenate([np.ones(len(s), dtype=int), np.zeros(len(b), dtype=int)])
    fpr, tpr, thr = roc_curve(y, -np.concatenate([s, b]))
    a = float(auc(fpr, tpr))
    k = int(np.argmax(tpr - fpr))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="firebrick", lw=2, label=f"Isolation Cut (AUC = {a:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random")
    ax.scatter([fpr[k]], [tpr[k]], color="firebrick", s=45, marker="s", edgecolors="black",
               zorder=5, label=f"Youden J = {tpr[k]-fpr[k]:.3f} (iso < {-thr[k]:.3f})")
    ax.set_xlabel("False Positive Rate (Non-Prompt Muons Misidentified as Prompt)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Prompt Muons Correctly Identified)", fontsize=10)
    ax.set_title(f"Isolation-Cut ROC - {name}, ΔR < {dr:g}", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"roc_dr{dr:g}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return a


#function to plot auc vs cone dR for all variants, with the run settings box
def summary_plot(rows, mode_names, o, min_pt):
    fig, ax = plt.subplots(figsize=(9.2, 6))
    order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    colors = [plt.cm.tab20(i / 19.0) for i in order]
    for j, name in enumerate(mode_names):
        sub = sorted([r for r in rows if r["mode"] == name], key=lambda r: r["dr"])
        ax.plot([r["dr"] for r in sub], [r["auc"] for r in sub], "o-", lw=2,
                color=colors[j], label=name)
    ax.set_xlabel("Isolation Cone ΔR", fontsize=12)
    ax.set_ylabel("Isolation-Cut AUC", fontsize=12)
    ax.set_title("Isolation-Cut AUC vs Cone ΔR for |Δz0sinθ| Window Table Variants", fontsize=12)
    ax.legend(title="Table Variant", fontsize=9, loc="center left",
              bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, frameon=True)
    ax.grid(alpha=0.3)
    variants = ", ".join(f"{m}: {MODE_DESC.get(m, m)}" for m in mode_names if m != "default")
    vlines = textwrap.wrap(variants, width=104) or [""]
    lines = [
        f"RUN SETTINGS   (isoPLOT_vary.py v{VERSION})",
        f"{'Samples':<12}: signal=run_InvPtPU200 (prompt), bkg=run_bjet (non-prompt); label=provenance",
        f"{'Variable':<12}: relative track isolation = ΣpT(cone tracks) / pT(muon)",
        f"{'Window':<12}: per-muon |Δz0sinθ| window from the (pT, |eta|) table",
        f"{'Variants':<12}: {vlines[0]}",
    ]
    lines += [f"{'':<12}  {x}" for x in vlines[1:]]
    lines += [
        f"{'Tracks':<12}: pT >= {min_pt/1000:.1f} GeV, muon excluded",
        f"{'Muons':<12}: signal {rows[0]['n_sig']:,} ({rows[0]['n_sig_ev']:,} events), "
        f"background {rows[0]['n_bkg']:,} ({rows[0]['n_bkg_ev']:,} events)",
    ]
    fig.text(0.01, -0.03, "\n".join(lines), ha="left", va="top", fontsize=8, family="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="whitesmoke", edgecolor="gray"))
    plt.tight_layout()
    plt.savefig(os.path.join(o, "iso_table_vary.png"), dpi=150, bbox_inches="tight")
    plt.close()


#function to regenerate the summary plot from an existing run's csv (--replot-dir)
def replot(d, min_pt):
    rows = []
    with open(os.path.join(d, "table_vary_summary.csv")) as fh:
        for r in csv.DictReader(fh):
            rows.append({"mode": r["mode"], "dr": float(r["dr"]), "auc": float(r["auc"]),
                         "n_sig": int(r["n_sig"]), "n_bkg": int(r["n_bkg"]),
                         "n_sig_ev": int(r.get("n_sig_ev") or 0),
                         "n_bkg_ev": int(r.get("n_bkg_ev") or 0)})
    seen = []
    for r in rows:
        if r["mode"] not in seen:
            seen.append(r["mode"])
    summary_plot(rows, seen, d, min_pt)
    print(f"replotted {os.path.join(d, 'iso_table_vary.png')}")


#function to parse command line arguments for sample paths, event count, dR cuts, and table variants
def parse_args():
    p = argparse.ArgumentParser(description="isolation-cut AUC vs auto-table variants")
    p.add_argument("--replot-dir", type=str, default=None,
                   help="regenerate the summary plot from an existing run's csv and exit")
    p.add_argument("--signal-path", type=str, default=isoPLOT.SIG_DEFAULT)
    p.add_argument("--bkg-path", type=str, default=isoPLOT.BKG_DEFAULT)
    p.add_argument("--n-events", type=int, default=0)
    p.add_argument("--min-pt", type=float, default=1000.0)
    p.add_argument("--dr-cuts", type=str, default="0.2,0.3,0.4,0.5")
    p.add_argument("--modes", type=str, default="default,eta-half,pt-half,pt-4th")
    return p.parse_args()


#main function
#isolation for both samples, then the table images, roc curves, csv, and summary plot
def main():
    args = parse_args()
    if args.replot_dir:
        replot(args.replot_dir, args.min_pt)
        return
    dr_cuts = [float(x) for x in args.dr_cuts.split(",")]
    mode_names = [m.strip() for m in args.modes.split(",")]
    for m in mode_names:
        if m not in MODES:
            raise SystemExit(f"unknown mode {m}; choose from {MODES}")
    variants = {m: make_variant(m) for m in mode_names}
    print(f"dR cuts: {dr_cuts}")
    print(f"modes:   {mode_names}")

    print("computing isolation for signal...")
    iso_s, n_sig_ev = isolations(args.signal_path, args.n_events, args.min_pt,
                                 dr_cuts, variants, True)
    print("computing isolation for background...")
    iso_b, n_bkg_ev = isolations(args.bkg_path, args.n_events, args.min_pt,
                                 dr_cuts, variants, False)

    #output directory under output/ (project root is three levels up from this file); suffix _runN avoids overwriting
    ev_tag = "MAX" if args.n_events <= 0 else str(args.n_events)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "output", f"iso_table_vary_v{VERSION}_n{ev_tag}")
    o = out
    i = 0
    while os.path.exists(o):
        i += 1
        o = f"{out}_run{i}"
    os.makedirs(o)

    #table images, the default table first as the reference
    tdir = os.path.join(o, "tables")
    os.makedirs(tdir)
    render_table("default_window", make_variant("default"), tdir)
    for name in mode_names:
        if name != "default":
            render_table(name, variants[name], tdir)
    print(f"wrote {len(mode_names) + (0 if 'default' in mode_names else 1)} table images")

    #roc curve per (variant, dR cut), auc collected into rows
    rows = []
    for name in mode_names:
        rdir = os.path.join(o, "roc", name)
        os.makedirs(rdir, exist_ok=True)
        for dr in dr_cuts:
            s = iso_s[(dr, name)]
            b = iso_b[(dr, name)]
            a = render_roc(name, dr, s, b, rdir)
            rows.append({"mode": name, "dr": dr, "auc": a,
                         "n_sig": len(s), "n_bkg": len(b)})
            print(f"  {name:<14} dR<{dr:<4} AUC = {a:.4f}")

    for r in rows:
        r["n_sig_ev"] = n_sig_ev
        r["n_bkg_ev"] = n_bkg_ev
    #write the auc table as csv (replot reads it back with --replot-dir)
    with open(os.path.join(o, "table_vary_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mode", "dr", "auc", "n_sig", "n_bkg",
                                           "n_sig_ev", "n_bkg_ev"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary_plot(rows, mode_names, o, args.min_pt)
    print(f"output dir: {o}")
    print("done")


#run main() only when executed as a script, not when imported
if __name__ == "__main__":
    main()
