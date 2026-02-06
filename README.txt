# ChronoEDMD ERSST v5 pipeline

This directory provides a reproducible pipeline to:
1) download NOAA ERSST v5 monthly NetCDF files,
2) merge them into a single file (ersst_v5_all.nc),
3) run the ChronoEDMD / Koopman workflow with LFO or LSO evaluation,
4) post-process logs and plot summary metrics.

Directory naming convention (recommended):
  <cv_mode>_<mm>_s<ss>
Example:
  lfo_08_s05
  lso_12_s03

------------------------------------------------------------
Requirements
------------------------------------------------------------

1) GNU parallel
   macOS:
     brew install parallel

2) Python environment
   The required Python packages depend on the scripts below
   (xarray, numpy, matplotlib, etc.).

------------------------------------------------------------
Files
------------------------------------------------------------

Data:
  ersst_v5_all.nc
    Merged ERSST v5 dataset used by the pipeline.

Download / build:
  download_ersst_v5_monthly.sh
    Download ERSST v5 monthly NetCDF files for a specified YYYYMM range.

  build_ersst_v5_all.py
    Merge monthly files (ersst.v5.YYYYMM.nc) into ersst_v5_all.nc.

Main pipeline:
  run_all.sh
    Batch run for cv_mode in (lfo|lso), fixed lead time s, for start months 1..12.

  run_lfo_all_s.sh
    LFO lead-time sweep for a fixed start month: s = 1..12.

  exec.sh
    Run a single experiment:
      ./exec.sh <smon:1..12> <cv_mode:lfo|lso> <s>
    This script calls the Python steps below and writes logs in the run directory.

Python steps called by exec.sh:
  make_sst_anom.py
  build_sig_kernel.py
  loo_koopman_eval.py
  plot_koopman_modes.py

Post-processing / plotting:
  plot_rms_kpc_with_clim.py
    Summarize RMS and kPC from logs and compare with climatology baselines.
  animate_mode.py
    Make an animation (animated GIF) of a Koopman mode as its phase varies.
    Run this inside a run directory (e.g., lfo_08_s05) where koopman_inputs_sig.npz exists.
------------------------------------------------------------
Workflow
------------------------------------------------------------

Step 1. Download ERSST v5 (example: match the paper period)
  ./download_ersst_v5_monthly.sh . 185401 202503

Step 2. Build ersst_v5_all.nc
  python -u build_ersst_v5_all.py

Step 3. Run batch experiments (example: LFO, s=5, jobs=2 default)
  ./run_all.sh lfo 5

Step 4. Optional: LFO lead-time sweep for a fixed start month (example: smon=8)
  ./run_lfo_all_s.sh 8 2

Step 5. Plot summaries
  python plot_rms_kpc_with_clim.py --cv_mode lfo --s 5

------------------------------------------------------------
Animation (mode phase)
------------------------------------------------------------

Create an animated GIF that shows the evolution of a selected mode
as the phase varies over [0, 2*pi].
Run this inside a run directory (e.g., lfo_08_s05) where koopman_inputs_sig.npz exists.

Usage:
  python -u ../animate_mode.py <mode_index>

Example:
  python -u ../animate_mode.py 23

------------------------------------------------------------
Outputs
------------------------------------------------------------

Each run directory (e.g., lfo_08_s05) contains:
  exec.log
  make_sst_anom_smonMM.log
  build_sig_kernel_smonMM.log
  loo_koopman_eval_<cv_mode>_smonMM_sS_*.log
  plot_koopman_modes_<cv_mode>_smonMM_sS.log
  koopman_inputs_sig.npz (required for plotting modes)

Notes:
- exec.sh creates a symlink to ../ersst_v5_all.nc inside each run directory.
- Avoid mixing different time ranges or datasets in the same run directory tree.
