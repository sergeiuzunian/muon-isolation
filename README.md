## Approaches to HL-LHC Prompt vs Non-Prompt Muon Track Discrimination

Sergei Uzunian\
Advisors: Jahred Adelman, Kevin Sedlaczek\
Northern Illinois University ATLAS Group\
US-ATLAS SUPER, Summer 2026

### Introduction

The Large Hadron Collider (LHC) is expected to begin Run 5 with pileup ⟨μ⟩ ≈ 200 around the late 2030's following the High Luminosity (HL) upgrade. The significant increase in simultaneous proton-proton collisions complicates the problem of distinguishing between prompt and non-prompt lepton tracks, including muons. Prompt leptons emerge from the decays of electroweak and Higgs bosons, whereas non-prompt muons tend to arise in hadron decays.

As a part of US-ATLAS SUPER, I have developed and compared the viability of an XGBoost Boosted Decision Tree (BDT) and a PyTorch Neural Network for prompt vs non-prompt muon track discrimination at HL-LHC pileup, operating on inner detector track information only: the reconstructed muon's own kinematics and impact parameters together with those of nearby tracks. While more sophisticated than standard isolation scalar cut-based discrimination (wherein relative isolation for a candidate muon is summed over a ΔR cone and scanned by threshold), these approaches remain considerably simpler than the transformer-based architectures developed for Runs 2 and 3, and outperform the isolation scalar threshold cut baseline.

### Principal Results

Following my advisor's guidance, the analysis is restricted to a physically motivated cone ΔR ≤ 0.5 (the jet scale), since the holdout AUC continues to rise with cone radius well beyond it in a manner that cannot reflect genuine isolation physics. At this cone the BDT reaches a holdout AUC of 0.8822, against 0.8604 for the Neural Network and 0.7740 for the best cut-based isolation.

Feature importances show that most of the discrimination is carried by the surrounding track activity rather than by the muon's own parameters. Consistent with this, the muon's own d0 gives a small gain (0.8822 with vs 0.8744 without), the neighboring-track d0 is negligible (0.8744 vs 0.8760 with d0 removed everywhere), and the optional isolation scalar and ζ-ordering add +0.000 and +0.0004 respectively, being already captured by the neighbor-track features. The improvement over the cut-based isolation is therefore attributable almost entirely to the models' access to the per-track kinematics of the cone (each neighbor's pT, ΔR, Δη and Δz0 sin θ), as opposed to a single pT sum taken behind a fixed |Δz0 sin θ| window.

### Samples

Training data was simulated for HL-LHC ⟨μ⟩ = 200 with a signal sample of single prompt muons with pileup and a background of non-prompt muons from b-hadron decays in HH(4b) events. Signal consisted of 9,291 events (one prompt muon per event) and background of 9,999 events with ∼1,970 tracks per event. Each event stores a list of reconstructed inner-detector tracks including per-track kinematics pT, η, φ and impact parameter values z0 sin θ (longitudinal) and d0 (transverse), with an `isMuon` flag identifying the reconstructed muons. Tracks below 1 GeV are dropped. The input ntuples are not distributed with this repository.

### Methodology

For the machine learning approaches, each candidate is a reconstructed muon. The candidate's feature vector consists of the muon's own pT, η, |z0 sin θ| and d0 together with a block of up to K = 10 neighboring tracks from a cone defined by ΔR and |Δz0 sin θ| about the candidate, each neighbor carrying its d0, pT, Δη, ΔR and Δz0 sin θ relative to the muon. Since the number of tracks inside the cone varies between events, the neighbor slots are filled in two orderings (ΔR-ascending and pT-descending), with unfilled slots NaN-padded for the BDT (handled natively by the XGBoost library) and zero-padded for the Neural Network.

Two kinds of reweighting are combined into one per-muon training weight. The first is kinematic: since the samples differ in their muon (pT, η) spectra, a gradient-boosted reweighter (`hep_ml`, fit on the training set only) weights the background so that its muon (pT, η) distribution matches signal, preventing separation on sample kinematics instead of isolation. The second corrects class imbalance, upweighting each signal muon by n_bkg/n_sig ≈ 62 (∼465k background vs ∼7.5k signal training muons).

The data is split 80/20 by event into training and holdout sets, and all quoted performance is the holdout AUC, with the training/holdout AUC gap used as the overfitting benchmark. Youden's J = TPR − FPR, maximized over the classification score threshold, provides a distinguished operating point weighting correct identification of prompt muons and rejection of non-prompt muons equally.

The Neural Network is a PyTorch multilayer perceptron of 105 → 15 → 15 → 1 with ReLU hidden layers and ∼1,850 parameters, trained with weighted cross-entropy and the Adam optimizer on the same inputs, preprocessing and split as the BDT, with the epoch of maximum holdout AUC taken as the stopping point.

The cut-based baseline computes relative track isolation Σ pT(cone) / pT(μ) over the same cone, scanned over ΔR and the |Δz0 sin θ| window for the best-performing isolation value, and serves as a proxy for any cut-based isolation tool. The window may alternatively be taken from a heuristic-motivated table developed by the NIU ATLAS group, which sets each nearby track's |Δz0 sin θ| requirement on the basis of its pT and η. Following a question asked at the CERN Isolation and Fakes Forum, the table's bin values were rescaled and the cone rescanned for each variant; dividing the pT bin values by three raised the best isolation AUC from 0.7510 to 0.7740, corresponding to a systematically wider |Δz0 sin θ| window than the original prescription.

### Limitations

Arguably the clearest limitation of the present work is the signal sample itself, in which the prompt muon is added directly without a simulated decay event. A natural next step is to train on samples in which prompt muons emerge from actual W, Z or Higgs decays, paired with a greater sample size.

The comparative performance metric used throughout this project is the area under the Receiver Operating Characteristic curve (ROC AUC), rather than non-prompt rejection at fixed prompt-efficiency working points, which would be the intended basis for eventual comparison against the isolation taggers used by ATLAS when more physical simulations of signal samples become available and the signal/background sample sizes are increased.

### Repository Layout

`main/` contains:

- `muon_iso_BDT/`: `muonBDT.py`, the XGBoost classifier script, with `run_muonbdt.pbs` and `run_isozeta.pbs`
- `muon_iso_NN/`: `muonNN.py` and `muonNN_small.py`, the PyTorch classifier scripts, with `run_muonnn.pbs`
- `muon_iso_cut/`: `isoPLOT.py` and `isoPLOT_vary.py`, the cut-based isolation scripts, with `run_isoplot.pbs` and `run_isovary.pbs`
- `plotting_tools/`: scripts which read run outputs (ROC overlays, rejection tables, tree-depth and network-architecture scans, neighbor-capture and sample-spectra plots) together with `summarize_runs.py`

Each subfolder of `main/` pairs its Python script with one batch wrapper PBS file. Every run writes a settings-named directory under `output/`, and `summarize_runs.py` collects them into dated `summary_tables` folders. Approximately 340 run directories are tracked. A standardized run-settings box is added to every plot and every run also writes a summary file, so that the output of any run can be traced to the settings which produced it.

`notebooks/` contains the original NIU ATLAS group introductory Jupyter notebook, `test_input_files.ipynb`, prior to the machine learning work.

`presentations/` contains:

- Slides from the CERN Isolation and Fakes Forum presentation, given July 27, 2026
- Slides from the US-ATLAS SUPER program symposium presentation, given August 20, 2026
- The Final Project Report for US-ATLAS SUPER

`ARCHIVE/` (mostly erroneous code prior to July 14, 2026) and `job_logs/` exist locally and are not tracked.

### Running the Code

The PBS wrappers were written for the NIU METIS cluster and record the configurations which were actually run. The BDT at the reported settings may be run directly as

```bash
python main/muon_iso_BDT/muonBDT.py --n-events 0 \
    --neighbor-dr-cut 0.5 --neighbor-dz-cut 15 --max-neighbors 10 \
    --d0-mode both --use-isolation true --use-gbreweighter true \
    --config "max_depth=3,n_estimators=200,learning_rate=0.10"
```

where `--n-events 0` selects all events. The Neural Network script accepts the same feature and reweighting arguments together with `--epochs`, `--hidden1` and `--hidden2`. The cut-based baseline is run as `isoPLOT.py --iso-dr-cut 0.4 --iso-dz-cut auto`, where `auto` selects the |Δz0 sin θ| window table.

The feature switches shared by the BDT and Neural Network scripts are as follows.

| Flag | Values | Description |
|---|---|---|
| `--d0-mode` | `both`, `muon-off`, `none` | Whether d0 is included for the muon and its neighbors, for the neighbors only, or for neither |
| `--use-neighbors` | `true`, `false` | Whether the K-neighbor block is included |
| `--use-isolation` | `true`, `false` | Whether the summed isolation scalar is included as an additional feature |
| `--use-zeta-order` | `true`, `false` | Whether a third neighbor ordering by ζ = √((20ΔR)² + (Δz0 sin θ)²) is included |
| `--use-gbreweighter` | `true`, `false` | Whether the kinematic (pT, η) reweighting is applied |

Dependencies: `uproot`, `awkward`, `numpy`, `xgboost`, `torch`, `scikit-learn`, `hep_ml`, `scipy`, `matplotlib`, `tqdm`.
