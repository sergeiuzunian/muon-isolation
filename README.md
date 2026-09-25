This repository includes code for undergraduate research conducted with the NIU ATLAS Group and the 2026 US-ATLAS SUPER REU program. 

We perform prompt vs non-prompt muon discrimination for the HL-LHC (pileup 200) on simulated reconstructed inner detector tracks, using track kinematics and impact parameters only (pT, eta, phi, z0sinθ, d0).

Signal data consists of single prompt muons injected into pileup (9291 events, one muon each), and background consists of muons from b-hadron decays in HH(4b) events (9999 events); each candidate muon is described by its own parameters as well as up to K=10 (value may be changed in future work) neighboring tracks in a dR and |Δz0sinθ| cone around it.

An XGBoost BDT and a small PyTorch MLP are trained on this feature vector (background reweighted to the signal muon (pT, eta), 80/20 split by event, holdout AUC only) and compared against a cut on relative track isolation summed over the same cone.

At the physical cone dR <= 0.5 the BDT and NN reach holdout AUC of 0.8822 and 0.8604 respectively. By contrast, the best scalar isolation threshold cut obtains an AUC of 0.7740, employing a numerically optimized eta and pT bin based longitudinal impact parameter cut.

## Repository Contents

main/ contains: 
- muon_iso_BDT (xgboost classifier script), 
- muon_iso_NN (pyTorch classifier script),
- muon_iso_cut (cut based isolation script),
- plotting_tools (scripts that read run outputs).

The classifier scripts, the isolation scripts, and the ROC overlay and rejection table plotting tools have batch wrapper pbs files next to them.

Every run writes a settings-named directory under output/; summarize_runs.py collects them into dated summary_tables folders.

notebooks/ contains:
- original NIU ATLAS group introductory information Jupyter Notebook (prior to machine learning work).

output/ contains:
- past outputs of BDT and Neural Network models and plotting tools (ROC plots, loss plots, feature importances, csv summaries, etc.).

presentations/ contains:
- Slides from CERN Isolation and Fakes Forum presentation, given July 27, 2026,
- Slides from US-ATLAS SUPER REU program symposium presentation, given August 20, 2026,
- Final Project Report for US-ATLAS SUPER REU.

For users with access to the METIS HPC project directory:
ARCHIVE/ (mostly erroneous code prior to 7/14/26) and job_logs/ exist locally but are not tracked. 

## How To Run

Python 3.12 venv with numpy, awkward, uproot, matplotlib, scipy, scikit-learn, xgboost, hep_ml, torch, pandas, tqdm (on METIS: `source ~/.venv/bin/activate`).

Samples: each script reads `<path>/OutputIsolation.root:OutputIsolation` with the branches InDetTrack_pt, InDetTrack_eta, InDetTrack_phi, InDetTrack_z0sinTheta, InDetTrack_d0, isMuon. Defaults point at run_InvPtPU200 (signal) and run_bjet (background) on /lstr/sahara; override with `--signal-path` and `--bkg-path`. `--n-events 0` loads everything (the nMAX runs in output/), the default 2000 is for quick tests.

Every run writes a settings-named directory under output/ (a _runN suffix is added instead of overwriting) with its plots, summary.csv and model. Every plot carries a run-settings box.

Batch: the classifier and isolation scripts, ROC overlay, and rejection table tools have a pbs wrapper next to them. Submit from the script's own directory (the wrapper cds to $PBS_O_WORKDIR) and create job_logs/ first. The job logs are not tracked on github.

```
mkdir -p job_logs
cd main/muon_iso_BDT && qsub run_muonbdt.pbs
```

BDT, best physical cone (main/muon_iso_BDT):

```
python muonBDT.py --n-events 0 --neighbor-dr-cut 0.5 --neighbor-dz-cut 10 --max-neighbors 10 --d0-mode both \
    --config "max_depth=3,n_estimators=200,learning_rate=0.10,reg_lambda=1,min_child_weight=1"
```

Repeat `--config` to fit several hyperparameter sets on one feature build (scan mode: output/muonbdt_scan_*/cfgNN_*/ plus scan_summary.csv). Feature switches: `--d0-mode both|muon-off|none`, `--use-isolation`, `--use-zeta-order`, `--use-neighbors`, `--use-gbreweighter`. Writes roc.png, importances.png, score and feature distributions, model.pkl, summary.csv.

NN, best configuration (main/muon_iso_NN), same cone and feature flags as the BDT:

```
python muonNN.py --n-events 0 --neighbor-dr-cut 0.5 --neighbor-dz-cut 10 --max-neighbors 10 --d0-mode both \
    --hidden1 15 --hidden2 15 --epochs 21 --batch-size 512 --learning-rate 0.001
```

Writes roc.png, loss_vs_epoch.png, auc_vs_epoch.png, model.pt, summary.csv (includes the best epoch).

Isolation cut (main/muon_iso_cut):

```
python isoPLOT.py --n-events 0 --iso-dr-cut 0.4 --iso-dz-cut auto        # dz: none, auto (pT,|eta| table) or a value in mm
python isoPLOT_vary.py --n-events 0 --dr-cuts 0.2,0.3,0.4,0.5 --modes default,pt-half,pt-3rd,pt-4th
python isoPLOT_vary.py --replot-dir ../../output/iso_table_vary_v1.0_nMAX   # redraw the summary plot from its csv
```

Plotting tools (main/plotting_tools) to run once the runs above exist:

```
python summarize_runs.py      # every nMAX summary.csv -> output/summary_tables_<date>/ (tables, heatmaps, best plots)
python bdt_depth_plot.py      # AUC vs tree depth from the dz10 d0on scans
python nn_arch_heatmap.py     # NN holdout AUC over hidden widths
python nn_best_epoch.py       # best epoch per NN configuration
python neighbor_capture.py --n-events 0 --dz-cut 15 --max-neighbors 10 20 30 40 50   # slot filling vs dR, plotted by summarize_runs.py
python sample_spectra.py      # track pT spectra of the two samples
python iso_bdt_roc.py         # isolation vs BDT ROC overlay
python nn_bdt_roc.py          # BDT vs NN ROC overlay
python iso_nn_bdt.py          # BDT vs NN vs isolation ROC overlay with Youden J
python nn_roc_youden.py       # best NN ROC with Youden J markers
python rejection_table.py     # non-prompt rejection at fixed prompt efficiency, rejection_table.csv
```
