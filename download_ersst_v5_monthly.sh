#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./download_ersst_v5_monthly.sh [OUTDIR] [START_YYYYMM] [END_YYYYMM]
#
# Example:
#   ./download_ersst_v5_monthly.sh . 185401 202503

OUTDIR="${1:-.}"
START_YYYYMM="${2:-185401}"
END_YYYYMM="${3:-202503}"

base="https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf"
mkdir -p "$OUTDIR"

FAILED_LIST="${OUTDIR%/}/failed_yyyymm.txt"
: > "$FAILED_LIST"   # truncate

is_yyyymm() {
  [[ "$1" =~ ^[0-9]{6}$ ]] || return 1
  local mm="${1:4:2}"
  [[ "$mm" -ge 1 && "$mm" -le 12 ]]
}
if ! is_yyyymm "$START_YYYYMM" || ! is_yyyymm "$END_YYYYMM" || [[ "$START_YYYYMM" -gt "$END_YYYYMM" ]]; then
  echo "Usage: $0 [OUTDIR] START_YYYYMM END_YYYYMM" >&2
  exit 2
fi

download_one () {
  local url="$1"
  local out="$2"
  local max_try="${3:-10}"
  local k=1

  while :; do
    # 失敗ログを一時ファイルへ
    tmp_err="$(mktemp)"
    if wget -4 --no-http-keep-alive --show-progress -c \
      --tries=1 --timeout=20 --read-timeout=20 --dns-timeout=10 \
      --retry-connrefused --waitretry=5 \
      -O "$out" "$url" 2> "$tmp_err"; then
      rm -f "$tmp_err"
      return 0
    fi

    # DNS失敗なら長めに待つ
    if grep -q "解決できませんでした\|nodename nor servname" "$tmp_err"; then
      echo "DNS failure detected; sleeping 60s..." >&2
      sleep 60
    else
      sleep 10
    fi
    rm -f "$tmp_err"

    if [[ "$k" -ge "$max_try" ]]; then
      return 1
    fi
    k=$((k+1))
  done
}

echo "ERSST v5 monthly download"
echo "  outdir: $OUTDIR"
echo "  range : $START_YYYYMM .. $END_YYYYMM"
echo

yyyymm="$START_YYYYMM"
while :; do
  year="${yyyymm:0:4}"
  month="${yyyymm:4:2}"
  f="ersst.v5.${year}${month}.nc"
  url="${base}/${f}"
  out="${OUTDIR%/}/${f}"

  # If already fully downloaded, skip quickly (avoid re-touching server)
  if [[ -f "$out" ]] && [[ "$(stat -f%z "$out" 2>/dev/null || echo 0)" -gt 100000 ]]; then
    echo "SKIP (exists): $f"
  else
    echo "GET: $f"
    if ! download_one "$url" "$out" 6 10; then
      echo "FAIL: $yyyymm $url" >&2
      echo "$yyyymm" >> "$FAILED_LIST"
    fi
    # be gentle
    sleep 0.2
  fi

  [[ "$yyyymm" == "$END_YYYYMM" ]] && break

  # increment month
  y=$((10#$year))
  m=$((10#$month))
  m=$((m + 1))
  if [[ "$m" -eq 13 ]]; then
    y=$((y + 1))
    m=1
  fi
  yyyymm=$(printf "%04d%02d" "$y" "$m")
done

echo
nf=$(ls -1 "$OUTDIR"/ersst.v5.??????.nc 2>/dev/null | wc -l | tr -d ' ')
echo "Downloaded files: $nf"
echo "Failed list     : $FAILED_LIST"
echo "Done."
