#!/usr/bin/env python3

import os
import csv
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

#project root is three levels up from this file (muonisolation/main/plotting_tools/)
OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output")
TABLES = os.path.join(OUTPUT, f"summary_tables_{datetime.now():%d.%m.%y}")


def read_summary(path):
    d = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) == 2 and row[0] != "key":
                d[row[0]] = row[1]
    d["_dir"] = os.path.basename(os.path.dirname(path))
    d["_path"] = path
    return d


def fnum(d, k, default=np.nan):
    try:
        return float(d[k])
    except (KeyError, ValueError, TypeError):
        return default


def d0_label(s):
    if "d0_mode" in s:
        return s["d0_mode"]
    v = s.get("use_d0")
    return "both" if v == "True" else ("none" if v == "False" else (v or ""))


def collect():
    bdt, iso = [], []
    for path in glob.glob(os.path.join(OUTPUT, "**", "summary.csv"), recursive=True):
        if "nMAX" not in path:
            continue
        s = read_summary(path)
        if "auc_holdout" in s and "use_isolation" in s and "muonnn_version" not in s:
            bdt.append({
                "version": s.get("muonbdt_version", "1.0*"),
                "dR": fnum(s, "neighbor_dr_cut"), "dz": fnum(s, "neighbor_dz_cut_mm"),
                "K": int(fnum(s, "max_neighbors_K", 0)),
                "d0": d0_label(s), "iso": s.get("use_isolation"), "reweight": s.get("use_gbreweighter"),
                "zeta": s.get("use_zeta_order", "False"),
                "depth": int(fnum(s, "max_depth", 0)), "trees": int(fnum(s, "n_estimators", 0)),
                "lr": fnum(s, "learning_rate"), "reg_lambda": fnum(s, "reg_lambda"),
                "mcw": fnum(s, "min_child_weight"), "n_feat": int(fnum(s, "n_features", 0)),
                "n_train": int(fnum(s, "n_train", 0)), "n_holdout": int(fnum(s, "n_holdout", 0)),
                "auc_train": fnum(s, "auc_train"), "auc_holdout": fnum(s, "auc_holdout"),
                "path": s["_path"],
            })
        elif "auc_isolation_cut" in s:
            iso.append({
                "version": s.get("isoplot_version", "1.0*"),
                "dR": fnum(s, "iso_dr_cut"), "dz": s.get("iso_dz_cut_mm", ""),
                "n_sig": int(fnum(s, "n_signal_muons", 0)), "n_bkg": int(fnum(s, "n_bkg_muons", 0)),
                "auc_iso": fnum(s, "auc_isolation_cut"), "youden_cut": fnum(s, "youden_iso_cut"),
                "path": s["_path"],
            })
    bdt = pd.DataFrame(bdt); iso = pd.DataFrame(iso)
    if not bdt.empty:
        bdt["gap"] = bdt["auc_train"] - bdt["auc_holdout"]
        bdt = bdt.sort_values("auc_holdout", ascending=False)
        bdt = bdt.drop_duplicates(
            subset=["dR", "dz", "K", "d0", "iso", "reweight", "zeta", "depth", "trees", "lr",
                    "reg_lambda", "mcw"], keep="first").reset_index(drop=True)
    if not iso.empty:
        iso = iso.sort_values("auc_iso", ascending=False)
        iso = iso.drop_duplicates(subset=["dR", "dz"], keep="first").reset_index(drop=True)
    return bdt, iso


HEADER = {
    "version": "Version", "dR": "ΔR", "dz": "dz", "K": "K", "d0": "d0", "iso": "Iso",
    "zeta": "ζ", "reweight": "Reweight", "depth": "Depth", "trees": "Trees", "lr": "LR",
    "reg_lambda": "Reg λ", "mcw": "MCW", "auc_train": "AUC Train", "auc_holdout": "AUC Holdout",
    "gap": "Gap", "n_sig": "N Sig", "n_bkg": "N Bkg", "auc_iso": "Isolation AUC",
    "youden_cut": "Youden Cut",
}


def render_table_image(df, cols, name, title, bw=False):
    disp = df[cols].copy()
    for c in disp.columns:
        if disp[c].dtype.kind == "f":
            disp[c] = disp[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
        else:
            disp[c] = disp[c].astype(str)
    nr, nc = disp.shape
    fig, ax = plt.subplots(figsize=(min(1.15 * nc + 0.5, 22), 0.34 * nr + 0.7))
    ax.axis("off")
    tbl = ax.table(cellText=disp.values, colLabels=[HEADER.get(c, c) for c in cols],
                   cellLoc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for j in range(nc):
        if bw:
            tbl[(0, j)].set_text_props(color="black", weight="bold")
        else:
            tbl[(0, j)].set_facecolor("#4472C4"); tbl[(0, j)].set_text_props(color="white", weight="bold")
            tbl[(1, j)].set_facecolor("#D4EDDA")
    ax.set_title(title, fontsize=14, weight="bold", pad=8)
    fig.savefig(os.path.join(TABLES, f"{name}.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def write_table(df, name, title, cols, bw=False):
    if df.empty:
        print(f"  ({name}: no runs found)"); return
    df.to_csv(os.path.join(TABLES, f"{name}.csv"), index=False)
    render_table_image(df, cols, name, title, bw=bw)
    print(f"  saved {name}.png / {name}.csv  ({len(df)} runs)")


D0_MODES = [("both", "d0both", "d0 On (Muon + Neighbor)"),
            ("muon-off", "d0muonoff", "Muon d0 Off (Neighbor d0 On)"),
            ("none", "d0none", "d0 Off (Muon + Neighbor)")]


def plot_bdt_cone_heatmap(bdt, d0_mode, suffix, phrase):
    g = bdt[(bdt["K"] == 10) & (bdt["reweight"] == "True") & (bdt["zeta"] == "False")
            & (bdt["d0"] == d0_mode) & (bdt["dR"] <= 1.0)]
    if g.empty:
        return
    drs = sorted(g["dR"].unique()); dzs = sorted(g["dz"].unique())
    M = np.full((len(drs), len(dzs)), np.nan)
    for i, dr in enumerate(drs):
        for j, dz in enumerate(dzs):
            sub = g[(g["dR"] == dr) & (g["dz"] == dz)]
            if not sub.empty:
                M[i, j] = sub["auc_holdout"].max()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(M, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(dzs))); ax.set_xticklabels([f"{d:g}" for d in dzs])
    ax.set_yticks(range(len(drs))); ax.set_yticklabels([f"{d:g}" for d in drs])
    ax.set_xlabel(r"Neighbor $|\Delta z_0\sin\theta|$ Cut [mm]"); ax.set_ylabel("Cone ΔR")
    ax.set_title(f"BDT Best Holdout AUC Over Cone (K=10, Reweight On)\n{phrase}")
    for i in range(len(drs)):
        for j in range(len(dzs)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.4f}", ha="center", va="center",
                        color="white" if M[i, j] < np.nanmax(M) - 0.01 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Holdout AUC"); fig.tight_layout()
    name = "bdt_cone_heatmap.png" if d0_mode == "both" else f"bdt_cone_heatmap_{suffix}.png"
    fig.savefig(os.path.join(TABLES, name), dpi=140); plt.close(fig)
    print(f"  saved {name}")


def plot_d0_vs_dr(bdt):
    g = bdt[(bdt["K"] == 10) & (bdt["reweight"] == "True") & (bdt["zeta"] == "False")]
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    drawn = False
    for d0_mode, _, phrase in D0_MODES:
        sub = g[g["d0"] == d0_mode].groupby("dR")["auc_holdout"].max().sort_index()
        if sub.empty:
            continue
        ax.plot(sub.index.values, sub.values, "o-", lw=2, label=phrase); drawn = True
    if not drawn:
        plt.close(fig); return
    ax.set_xlabel("Cone ΔR"); ax.set_ylabel("Holdout AUC")
    ax.set_title(r"BDT Holdout AUC vs Cone ΔR by d0 Mode (Best $|\Delta z_0\sin\theta|$, Reweight On)")
    ax.grid(alpha=0.3); ax.legend(title="d0 Mode", loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(TABLES, "bdt_d0_vs_dr.png"), dpi=140); plt.close(fig)
    print("  saved bdt_d0_vs_dr.png")


def plot_bdt_cone_heatmap_dr5(bdt):
    short = {"both": "Both (μ+nbr)", "muon-off": "Muon Off", "none": "None"}
    g = bdt[(bdt["K"] == 10) & (bdt["reweight"] == "True") & (bdt["zeta"] == "False")
            & (bdt["dz"] == 15)]
    if g.empty:
        return
    modes = [m for m, _, _ in D0_MODES if not g[g["d0"] == m].empty]
    drs = sorted(g["dR"].unique())
    M = np.full((len(drs), len(modes)), np.nan)
    for i, dr in enumerate(drs):
        for j, m in enumerate(modes):
            sub = g[(g["dR"] == dr) & (g["d0"] == m)]
            if not sub.empty:
                M[i, j] = sub["auc_holdout"].max()
    fig, ax = plt.subplots(figsize=(6.5, 9))
    im = ax.imshow(M, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(modes))); ax.set_xticklabels([short[m] for m in modes])
    ax.set_yticks(range(len(drs))); ax.set_yticklabels([f"{d:g}" for d in drs])
    ax.set_xlabel("d0 Mode"); ax.set_ylabel("Cone ΔR")
    ax.set_title("BDT Holdout AUC vs Cone ΔR to 5.0 by d0 Mode\n($|\\Delta z_0\\sin\\theta| < 15$ mm, Reweight On)")
    hi = np.nanmax(M)
    for i in range(len(drs)):
        for j in range(len(modes)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.4f}", ha="center", va="center",
                        color="white" if M[i, j] < hi - 0.05 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Holdout AUC"); fig.tight_layout()
    fig.savefig(os.path.join(TABLES, "bdt_cone_heatmap_dr_to5.png"), dpi=140); plt.close(fig)
    print("  saved bdt_cone_heatmap_dr_to5.png")


def _render_capture(sub, K, dz, out_name):
    style = {"signal": ("steelblue", "Prompt Muon (Signal)"),
             "background": ("goldenrod", "Non-Prompt Muon (Background)")}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for name, (color, lab) in style.items():
        g = sub[sub["sample"] == name].sort_values("dr")
        ax[0].plot(g["dr"], g["mean_filled_slots"], "o-", color=color, lw=2, label=lab)
        ax[1].plot(g["dr"], g["frac_saturated"], "o-", color=color, lw=2, label=lab)
    ax[0].axhline(K, color="gray", ls=":", lw=1)
    ax[0].set_ylabel(f"Mean Filled Neighbor Slots (of K = {K})")
    ax[0].set_title("Captured Neighbor Tracks vs Cone ΔR")
    ax[1].set_ylabel(f"Fraction of Muons With All K = {K} Slots Filled")
    ax[1].set_title("Neighbor Slot Saturation vs Cone ΔR")
    for a in ax:
        a.axvline(0.5, color="gray", ls="--", lw=1, alpha=0.7, label="Physical Cone Limit (ΔR = 0.5)")
        a.set_xlabel("Cone ΔR"); a.grid(alpha=0.3); a.legend(fontsize=9, loc="lower right")
    fig.suptitle(rf"BDT Neighbor Capture (K = {K}, $|\Delta z_0\sin\theta| < {dz:g}$ mm)", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(TABLES, out_name), dpi=140); plt.close(fig)
    print(f"  saved {out_name}")


def plot_neighbor_capture():
    src = os.path.join(OUTPUT, "neighbor_capture", "neighbor_capture.csv")
    if not os.path.exists(src):
        return
    df = pd.read_csv(src)
    dz = df["dz_cut"].iloc[0]
    for K in sorted(df["K"].unique()):
        sub = df[df["K"] == K]
        _render_capture(sub, int(K), dz, f"bdt_neighbor_capture_K{int(K)}.png")
        if int(K) == 10:
            _render_capture(sub, 10, dz, "bdt_neighbor_capture.png")


def plot_bdt_depth(bdt, d0_mode, suffix, phrase):
    g = bdt[(bdt["K"] == 10) & (bdt["reweight"] == "True") & (bdt["zeta"] == "False")
            & (bdt["d0"] == d0_mode)].copy()
    if g.empty:
        return
    g = g[g.groupby(["dR", "dz"])["depth"].transform("nunique") >= 4]
    if g.empty:
        print(f"  (bdt_auc_vs_depth {d0_mode}: no cone has >=4 depths yet)")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for (dr, dz), sub in g.groupby(["dR", "dz"]):
        best = sub.groupby("depth")["auc_holdout"].max().sort_index()
        ax.plot(best.index.values, best.values, "o-", lw=2, label=rf"ΔR < {dr:g}, $|\Delta z_0\sin\theta| < {dz:g}$ mm")
    ax.set_xlabel("Max Tree Depth"); ax.set_ylabel("Holdout AUC")
    ax.set_title(f"BDT Holdout AUC vs Tree Depth\n{phrase}")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, title="Cone", loc="lower left")
    name = "bdt_auc_vs_depth.png" if d0_mode == "both" else f"bdt_auc_vs_depth_{suffix}.png"
    fig.tight_layout(); fig.savefig(os.path.join(TABLES, name), dpi=140); plt.close(fig)
    print(f"  saved {name}")


def plot_iso_heatmap(iso):
    if iso.empty:
        return
    drs = sorted(iso["dR"].unique()); dzs = sorted(iso["dz"].unique(), key=str)
    M = np.full((len(drs), len(dzs)), np.nan)
    for i, dr in enumerate(drs):
        for j, dz in enumerate(dzs):
            sub = iso[(iso["dR"] == dr) & (iso["dz"] == dz)]
            if not sub.empty:
                M[i, j] = sub["auc_iso"].max()
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(M, cmap="magma", aspect="auto", origin="lower")
    ax.set_xticks(range(len(dzs))); ax.set_xticklabels([str(d) for d in dzs])
    ax.set_yticks(range(len(drs))); ax.set_yticklabels([f"{d:g}" for d in drs])
    ax.set_xlabel(r"$|\Delta z_0\sin\theta|$ Cut"); ax.set_ylabel("Isolation Cone ΔR")
    ax.set_title("Isolation-Cut AUC Over Cone")
    hi = np.nanmax(M)
    for i in range(len(drs)):
        for j in range(len(dzs)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.4f}", ha="center", va="center",
                        color="white" if M[i, j] < hi - 0.05 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Isolation AUC"); fig.tight_layout()
    fig.savefig(os.path.join(TABLES, "iso_cone_heatmap.png"), dpi=140); plt.close(fig)
    print("  saved iso_cone_heatmap.png")


def plot_comparison(bdt, iso):
    if bdt.empty or iso.empty:
        return
    bb = bdt["auc_holdout"].max(); bi = iso["auc_iso"].max()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Isolation Cut\n(Best Cone)", "BDT\n(Best Config)"], [bi, bb],
           color=["firebrick", "steelblue"], edgecolor="black")
    for x, v in enumerate([bi, bb]):
        ax.text(x, v + 0.005, f"{v:.4f}", ha="center", fontsize=12)
    ax.set_ylabel("AUC (Prompt vs Non-Prompt)"); ax.set_ylim(0.5, 1.0)
    ax.set_title(f"Best Isolation vs Best BDT  (Δ = {bb-bi:+.3f})")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(os.path.join(TABLES, "best_iso_vs_bdt.png"), dpi=140); plt.close(fig)
    print("  saved best_iso_vs_bdt.png")


def plot_comparison_physical(bdt, iso):
    if bdt.empty or iso.empty:
        return
    g = bdt[(bdt["reweight"] == "True") & (bdt["zeta"] == "False") & (bdt["dR"] <= 0.5)]
    if g.empty:
        return
    bb = g["auc_holdout"].max(); bi = iso["auc_iso"].max()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Isolation Cut\n(Best Cone)", "BDT\n(Best, ΔR ≤ 0.5)"], [bi, bb],
           color=["firebrick", "steelblue"], edgecolor="black")
    for x, v in enumerate([bi, bb]):
        ax.text(x, v + 0.005, f"{v:.4f}", ha="center", fontsize=12)
    ax.set_ylabel("AUC (Prompt vs Non-Prompt)"); ax.set_ylim(0.5, 1.0)
    ax.set_title(f"Best Isolation vs Best Physical-Cone BDT  (Δ = {bb-bi:+.3f})")
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
    fig.savefig(os.path.join(TABLES, "best_iso_vs_bdt_physical.png"), dpi=140); plt.close(fig)
    print("  saved best_iso_vs_bdt_physical.png")


def _copy_plot(row, fname, dest):
    src = os.path.join(os.path.dirname(row["path"]), fname)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(TABLES, dest)); print(f"  copied {dest}")
    else:
        print(f"  (missing {fname} for {dest})")


def copy_best_plots(bdt, iso):
    if not bdt.empty:
        _copy_plot(bdt.iloc[0], "roc.png", "best_bdt_overall_roc.png")
        gp = bdt[(bdt["reweight"] == "True") & (bdt["zeta"] == "False") & (bdt["dR"] <= 0.5)]
        for d0_mode, suffix, _ in D0_MODES:
            sub = gp[gp["d0"] == d0_mode].sort_values("auc_holdout", ascending=False)
            if sub.empty:
                continue
            row = sub.iloc[0]
            _copy_plot(row, "roc.png", f"best_bdt_physical_{suffix}_roc.png")
            _copy_plot(row, "importances.png", f"best_bdt_physical_{suffix}_importances.png")
    if not iso.empty:
        phys = iso[iso["dz"] != "none"].sort_values("auc_iso", ascending=False)
        if not phys.empty:
            _copy_plot(phys.iloc[0], "roc.png", "best_iso_physical_roc.png")
        naive = iso[iso["dz"] == "none"].sort_values("auc_iso", ascending=False)
        if not naive.empty:
            _copy_plot(naive.iloc[0], "roc.png", "best_iso_nodz_roc.png")


def main():
    os.makedirs(TABLES, exist_ok=True)
    bdt, iso = collect()
    print(f"found {len(bdt)} BDT runs, {len(iso)} isolation runs (nMAX, current version)")
    bdt_cols = ["version", "dR", "dz", "K", "d0", "iso", "zeta", "reweight", "depth", "trees", "lr",
                "reg_lambda", "mcw", "auc_train", "auc_holdout", "gap"]
    iso_cols = ["version", "dR", "dz", "n_sig", "n_bkg", "auc_iso", "youden_cut"]
    if not bdt.empty:
        write_table(bdt, "bdt_runs", "BDT Runs (Max Events) — Sorted by Holdout AUC", bdt_cols, bw=True)
    if not iso.empty:
        write_table(iso, "iso_runs", "Isolation-Cut Runs (Max Events) — Sorted by AUC", iso_cols, bw=True)
    if not bdt.empty:
        print(f"  best BDT:  holdout {bdt.iloc[0]['auc_holdout']:.4f}  "
              f"(ΔR<{bdt.iloc[0]['dR']:g}, dz<{bdt.iloc[0]['dz']:g}, K{bdt.iloc[0]['K']}, "
              f"depth{bdt.iloc[0]['depth']}, reweight={bdt.iloc[0]['reweight']})")
        for d0_mode, suffix, phrase in D0_MODES:
            plot_bdt_cone_heatmap(bdt, d0_mode, suffix, phrase)
            plot_bdt_depth(bdt, d0_mode, suffix, phrase)
        plot_d0_vs_dr(bdt); plot_bdt_cone_heatmap_dr5(bdt)
    if not iso.empty:
        print(f"  best iso:  AUC {iso.iloc[0]['auc_iso']:.4f}  "
              f"(ΔR<{iso.iloc[0]['dR']:g}, dz={iso.iloc[0]['dz']})")
        plot_iso_heatmap(iso)
    plot_comparison(bdt, iso); plot_comparison_physical(bdt, iso)
    plot_neighbor_capture()
    copy_best_plots(bdt, iso)
    print(f"\nall outputs in {TABLES}")


if __name__ == "__main__":
    main()
