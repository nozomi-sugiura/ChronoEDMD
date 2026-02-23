#!/usr/bin/env bash
set -euo pipefail

python -u ../calc_tpi.py "$1"

mode3=$(printf "%03d" "$2")
python -u ../tpi_mode_pipeline.py --use raw --bandpass --p_low 21 --p_high 35 \
       --mode_csv "frames/timeseries_${mode3}.csv" --use "raw"


