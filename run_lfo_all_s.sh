#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# ---------------------------------------------
# run_lfo_all_s.sh
#
# Run LFO for a fixed start-month (smon) and sweep lead time s=1..12.
# If the target directory already exists, skip that (s) to avoid overwrite.
#
# usage:
#   bash ./run_lfo_all_s.sh <smon:1..12> [jobs]
# example:
#   bash ./run_lfo_all_s.sh 8 2
# ---------------------------------------------

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <smon:1..12> [jobs]" >&2
  exit 1
fi

m="$1"
jobs="${2:-2}"

if ! [[ "$m" =~ ^[0-9]+$ ]] || (( m < 1 || m > 12 )); then
  echo "smon must be integer 1..12" >&2
  exit 1
fi
if ! [[ "$jobs" =~ ^[0-9]+$ ]] || (( jobs < 1 )); then
  echo "jobs must be positive integer" >&2
  exit 1
fi

# Avoid CPU oversubscription from BLAS/OpenMP
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if ! command -v parallel >/dev/null 2>&1; then
  echo "ERROR: GNU parallel is required (brew install parallel)." >&2
  exit 1
fi
if [[ ! -x ./exec.sh ]]; then
  echo "ERROR: ./exec.sh not found or not executable. Run: chmod +x exec.sh" >&2
  exit 1
fi

m2="$(printf "%02d" "$m")"
echo "LFO lead-time sweep for smon=$m (dir tag=$m2), jobs=$jobs"
echo "Skip existing directories."
echo

seq 1 12 | parallel -j "$jobs" --line-buffer '
  s={}
  s2=$(printf "%02d" "$s")
  dir="lfo_'$m2'_s${s2}"

  if [[ -d "$dir" ]]; then
    echo "# SKIP ${dir} (already exists)"
    exit 0
  fi

  mkdir -p "$dir"
  (
    set -euo pipefail
    cd "$dir"
    echo "# START smon='$m' cv_mode=lfo s=$s $(date -Iseconds)"
    ../exec.sh "'$m'" lfo "$s"
    echo "# END   smon='$m' cv_mode=lfo s=$s $(date -Iseconds)"
  ) > exec.log 2>&1

  echo "# DONE ${dir} (log: ${dir}/exec.log)"
'

echo
echo "Done."
