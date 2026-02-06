#!/usr/bin/env bash
set -euo pipefail
# --- prevent BLAS/OMP oversubscription (important on macOS) ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# ---------------------------------------------
# exec.sh
#
# usage:
#   bash ./exec.sh <smon:1..12> <cv_mode:lfo|lso> <s>
# example:
#   bash ./exec.sh 8 lfo 5
# ---------------------------------------------

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <smon:1..12> <cv_mode:lfo|lso> <s>"
  exit 1
fi

smon="$1"
cv_mode="$2"
s="$3"

lmda_k="2.0"
lmda_tag="${lmda_k/./p}"   # 2.0 -> 2p0

case "$cv_mode" in
  lfo|lso) ;;
  *) echo "cv_mode must be lfo or lso"; exit 1 ;;
esac

# dataset symlink (local dir)
ln -sf ../ersst_v5_all.nc .
echo "# linked dataset: $(readlink ersst_v5_all.nc || echo ersst_v5_all.nc)"

# optional: zero-pad for tidy filenames
smon2="$(printf "%02d" "$smon")"

python -u ../make_sst_anom.py "$smon" | tee "make_sst_anom_smon${smon2}.log"
python -u ../build_sig_kernel.py      | tee "build_sig_kernel_smon${smon2}.log"

python -u ../loo_koopman_eval.py \
  --cv_mode "$cv_mode" --kernel both --s "$s" --ord 7 \
  --min_rej 0 --max_rej 20 --rej_step 4 --lmda_1d_maxiter 12 --optimize_sig \
  --xfeat concat12 --vmax_rms 1 --vmax_drms 0.075 \
  --lmda_kpc "$lmda_k" --save_inputs \
  | tee "loo_koopman_eval_${cv_mode}_smon${smon2}_s${s}_lmda${lmda_tag}.log"

# safety: ensure the expected saved input exists before plotting
[[ -f "koopman_inputs_sig.npz" ]] || { echo "ERROR: koopman_inputs_sig.npz not found"; exit 1; }

python -u ../plot_koopman_modes.py "$smon" "koopman_inputs_sig.npz" \
    | tee "plot_koopman_modes_${cv_mode}_smon${smon2}_s${s}.log"
