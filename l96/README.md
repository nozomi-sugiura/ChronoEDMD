# Two-scale Lorenz-96 benchmark

This directory contains scripts for the two-scale Lorenz-96 benchmark used to compare path-based Sig-EDMD, SPK, Kiraly-type signature kernels, and Snap-DMD.

## Files

- `model_l96_2scale_argF.py`  
  Generates slow-variable trajectories of the two-scale Lorenz-96 model with user-specified forcing `F`.

- `calc_sig_vs_kiraly_spk_l96_with_stats.py`  
  Runs the path-based Sig-EDMD, SPK, and Kiraly-type signature-kernel experiments.

- `calc_snapshot_edmd_l96_with_stats.py`  
  Runs the Snap-DMD baseline.

- `plot_ensemble_stats_model_time_with_snap_Fdirs.py`  
  Plots ensemble-mean prediction skill curves.

- `run_seeds.sh`  
  Runs the full benchmark for seeds 0--9 by default.

## Example

```bash
./run_seeds.sh 8.0
