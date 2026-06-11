#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_seeds_F_default_0_9_with_snapshot.sh F_VALUE
#   bash run_seeds_F_default_0_9_with_snapshot.sh F_VALUE SEED1 [SEED2 ...]
#
# Examples:
#   bash run_seeds_F_default_0_9_with_snapshot.sh 8.0
#   bash run_seeds_F_default_0_9_with_snapshot.sh 10.0 0 1 2 3

if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 F_VALUE [SEED1 SEED2 ...]" >&2
  echo "Example: bash $0 8.0" >&2
  echo "Example: bash $0 8.0 0 1 2 3" >&2
  exit 1
fi

F_VALUE="$1"
shift

# Default seed ensemble: 0,1,...,9
if [[ $# -eq 0 ]]; then
  SEEDS=(0 1 2 3 4 5 6 7 8 9)
else
  SEEDS=("$@")
fi

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

F_TAG=$(printf "%s" "$F_VALUE" | sed 's/-/m/g; s/\./p/g')

echo "F_VALUE = $F_VALUE"
echo "SEEDS   = ${SEEDS[*]}"
echo ""

for seed in "${SEEDS[@]}"; do
  seed_pad=$(printf "%03d" "$seed")
  workdir="F${F_TAG}_S${seed_pad}"

  echo "============================================================"
  echo "F       = $F_VALUE"
  echo "SEED    = $seed"
  echo "WORKDIR = $workdir"
  echo "============================================================"

  mkdir -p "$workdir"

  (
    cd "$workdir"
    mkdir -p data

    echo "[1/3] Running model_l96_2scale_argF.py with F=$F_VALUE, SEED=$seed"
    python -u "$MODEL_SCRIPT" "$seed" --F "$F_VALUE" --data-file "data/l96.npz" \
      2>&1 | tee "model_l96_2scale_F${F_TAG}_seed${seed_pad}.log"

    echo "[2/3] Running calc_sig_vs_kiraly_spk_l96_with_stats.py"
    python -u "$SIG_SCRIPT" \
      2>&1 | tee "calc_sig_vs_kiraly_spk_l96_F${F_TAG}_seed${seed_pad}.log"

    echo "[3/3] Running calc_snapshot_edmd_l96_with_stats.py"
    python -u "$SNAPSHOT_SCRIPT" \
      2>&1 | tee "calc_snapshot_edmd_l96_F${F_TAG}_seed${seed_pad}.log"
  )

  echo "Finished F=$F_VALUE, seed=$seed. Results are in $workdir/"
  echo ""
done

echo "All seeds finished for F=$F_VALUE."
