#!/usr/bin/env bash
set -euo pipefail

# Run the three L96 cases used in the paper.
#
# Cases:
#   A: F=10, h=0.50
#   B: F=10, h=0.75
#   C: F=10, h=1.00
#
# Usage:
#   bash run_seeds_ABC_F10_h_cases.sh
#   bash run_seeds_ABC_F10_h_cases.sh SEED1 [SEED2 ...]
#
# Examples:
#   bash run_seeds_ABC_F10_h_cases.sh
#   bash run_seeds_ABC_F10_h_cases.sh 0 1 2 3

F_VALUE="10.0"

# Default seed ensemble: 0,1,...,9
if [[ $# -eq 0 ]]; then
  SEEDS=(0 1 2 3 4 5 6 7 8 9)
else
  SEEDS=("$@")
fi

# Case labels and h values must match the paper.
CASES=("A:0.50" "B:0.75" "C:1.00")

ROOT_DIR="$(pwd)"

MODEL_SCRIPT="${ROOT_DIR}/../model_l96_2scale_argF.py"
SIG_SCRIPT="${ROOT_DIR}/../calc_sig_vs_kiraly_spk_l96_with_stats.py"
SNAPSHOT_SCRIPT="${ROOT_DIR}/../calc_snapshot_edmd_l96_with_stats.py"

for script in "$MODEL_SCRIPT" "$SIG_SCRIPT" "$SNAPSHOT_SCRIPT"; do
  if [[ ! -f "$script" ]]; then
    echo "ERROR: script not found: $script" >&2
    exit 1
  fi
done

tag_value() {
  printf "%s" "$1" | sed 's/-/m/g; s/\./p/g'
}

F_TAG=$(tag_value "$F_VALUE")

echo "F_VALUE = $F_VALUE"
echo "CASES   = ${CASES[*]}"
echo "SEEDS   = ${SEEDS[*]}"
echo ""

for case_item in "${CASES[@]}"; do
  CASE_LABEL="${case_item%%:*}"
  H_VALUE="${case_item#*:}"
  H_TAG=$(tag_value "$H_VALUE")

  for seed in "${SEEDS[@]}"; do
    seed_pad=$(printf "%03d" "$seed")
    workdir="case${CASE_LABEL}_F${F_TAG}_h${H_TAG}_S${seed_pad}"

    echo "============================================================"
    echo "CASE    = $CASE_LABEL"
    echo "F       = $F_VALUE"
    echo "h       = $H_VALUE"
    echo "SEED    = $seed"
    echo "WORKDIR = $workdir"
    echo "============================================================"

    mkdir -p "$workdir"

    (
      cd "$workdir"
      mkdir -p data

      echo "[1/3] Running model_l96_2scale_argF.py with F=$F_VALUE, h=$H_VALUE, SEED=$seed"
      python -u "$MODEL_SCRIPT" "$seed" --F "$F_VALUE" --h "$H_VALUE" --data-file "data/l96.npz" \
        2>&1 | tee "model_l96_2scale_case${CASE_LABEL}_F${F_TAG}_h${H_TAG}_seed${seed_pad}.log"

      echo "[2/3] Running calc_sig_vs_kiraly_spk_l96_with_stats.py"
      python -u "$SIG_SCRIPT" \
        2>&1 | tee "calc_sig_vs_kiraly_spk_l96_case${CASE_LABEL}_F${F_TAG}_h${H_TAG}_seed${seed_pad}.log"

      echo "[3/3] Running calc_snapshot_edmd_l96_with_stats.py"
      python -u "$SNAPSHOT_SCRIPT" \
        2>&1 | tee "calc_snapshot_edmd_l96_case${CASE_LABEL}_F${F_TAG}_h${H_TAG}_seed${seed_pad}.log"
    )

    echo "Finished case=$CASE_LABEL, F=$F_VALUE, h=$H_VALUE, seed=$seed. Results are in $workdir/"
    echo ""
  done
done

echo "All cases and seeds finished."
