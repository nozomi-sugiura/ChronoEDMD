#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"   # スクリプトのあるディレクトリで実行

python3 build_ersst_v5_all.py

# 出力ファイルが「存在」し「サイズ>0」なら削除
test -s ersst_v5_all.nc

# 念のため、削除対象が実際に存在することを確認してから削除（空展開事故防止）
shopt -s nullglob
files=(ersst.v5.??????.nc)
if ((${#files[@]}==0)); then
  echo "No monthly files to delete (ersst.v5.??????.nc)."
  exit 0
fi

rm -f -- "${files[@]}"
echo "Deleted ${#files[@]} monthly files."
