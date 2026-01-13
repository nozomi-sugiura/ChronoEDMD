#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <smon:1..12> <cv_mode:lfo|lso>"
  exit 1
fi

smon="$1"
cv_mode="$2"
lmda_k="2.0"
lmda_tag="${lmda_k/./p}"   # 2.0 -> 2p0

case "$cv_mode" in
  lfo|lso) ;;
  *) echo "cv_mode must be lfo or lso"; exit 1 ;;
esac

ln -sf ./ersst_v5_all.nc .

python -u ./make_sst_anom.py "$smon" | tee "make_sst_anom_smon${smon}.log"
python -u ./build_sig_kernel.py       | tee "build_sig_kernel_smon${smon}.log"

python -u ./loo_koopman_eval.py \
  --cv_mode "$cv_mode" --kernel both --s 5 --ord 7 \
  --min_rej 15 --max_rej 20 --rej_step 1 --optimize_sig \
  --xfeat concat12 --vmax_rms 1 --vmax_drms 0.05 \
  --lmda_kpc "$lmda_k" --save_inputs \
  | tee "loo_koopman_eval_${cv_mode}_smon${smon}_lmda${lmda_tag}.log"

python -u ./plot_koopman_modes.py "$smon" "koopman_inputs_sig.npz" \
    | tee "plot_koopman_modes_${cv_mode}_smon${smon}.log"
