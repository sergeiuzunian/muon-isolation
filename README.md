main/ contains: 
- muon_iso_BDT (xgboost classifier script), 
- muon_iso_NN (pyTorch classifier script),
- muon_iso_cut (cut based isolation script),
- plotting_tools (scripts that read run outputs).

Each subfolder of main pairs its python script with one batch wrapper pbs file.

Every run writes a settings-named directory under output/;summarize_runs.py collects them into dated summary_tables folders.

presentations/ contains:
- Slides from CERN Isolation and Fakes Forum presentation, given July 27, 2026,
- Slides from US-ATLAS SUPER REU program symposium presentation, given August 20, 2026,
- Final Project Report for US-ATLAS SUPER REU.

ARCHIVE/ (mostly erroneous code prior to 7/14/26) and job_logs/ exist locally but are not tracked. 
