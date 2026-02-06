#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_all.sh (lfo|lso) <s> [N_JOBS]
#
# Examples:
#   ./run_all.sh lfo 5
#   ./run_all.sh lso 5 2
#   ./run_all.sh lfo 3 4

MODE="${1:-}"
S="${2:-}"
NJOBS="${3:-2}"

if [[ -z "$MODE" || -z "$S" ]]; then
  echo "Usage: $0 (lfo|lso) <s> [N_JOBS]" >&2
  exit 2
fi
if [[ "$MODE" != "lfo" && "$MODE" != "lso" ]]; then
  echo "Invalid MODE: $MODE (must be lfo or lso)" >&2
  exit 2
fi
if ! [[ "$S" =~ ^[0-9]+$ ]] || (( S < 1 )); then
  echo "Invalid s: $S (must be positive integer)" >&2
  exit 2
fi
if ! [[ "$NJOBS" =~ ^[0-9]+$ ]] || (( NJOBS < 1 )); then
  echo "Invalid N_JOBS: $NJOBS (must be positive integer)" >&2
  exit 2
fi

# --- deps ---
if ! command -v parallel >/dev/null 2>&1; then
  echo "ERROR: GNU parallel is required (brew install parallel)." >&2
  exit 1
fi

# --- locate exec.sh ---
if [[ ! -x ./exec.sh ]]; then
  echo "ERROR: ./exec.sh not found or not executable. Run: chmod +x exec.sh" >&2
  exit 1
fi

s2="$(printf "%02d" "$S")"

echo "Run mode: $MODE"
echo "s       : $S (tag s${s2})"
echo "Jobs    : $NJOBS"
echo

months=$(seq -w 1 12)

# Create dirs: lfo_01_s05 ... lfo_12_s05
for mm in $months; do
  mkdir -p "${MODE}_${mm}_s${s2}"
done

# Run in parallel:
# - directory name gives mm
# - smon = integer(mm)
# - call: ../exec.sh smon MODE S
parallel -j "$NJOBS" --halt now,fail=1 --line-buffer \
  'd="{1}"; mm="${d#*_}"; mm="${mm%_s*}"; smon=$((10#${mm})); cd "$d" && ../exec.sh "$smon" "'"$MODE"'" "'"$S"'" > exec.log 2>&1' ::: \
  $(for mm in $months; do echo "${MODE}_${mm}_s${s2}"; done)

echo
echo "Done: ${MODE}_01_s${s2} .. ${MODE}_12_s${s2}"
