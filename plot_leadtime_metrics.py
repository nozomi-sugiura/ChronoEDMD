#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import glob
import numpy as np
import matplotlib.pyplot as plt
LOG_GLOB = "lfo_08_s*/loo*log"
re_lead = re.compile(r"lfo_08_s=?(\d{2})/")
leads = np.arange(1, 13, dtype=int)
#LOG_GLOB = "lfo_08_s=*/loo*log"
ASSUME_RMS_1ST_SPK_2ND_SIG = True

OUT_KPC = "leadtime_kPC.pdf"
OUT_RMS = "leadtime_RMS.pdf"

#re_lead = re.compile(r"lfo_08_s=(\d+)/")

NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

# kPC
re_kpc = re.compile(
    rf"\[(SIG|SPK)\s+kPC\]\s+mean_kPC_(model|clim)\s*=\s*({NUM})\s+"
    rf"std_kPC_(model|clim)\s*=\s*({NUM})"
)

# RMS: kernel が SIG/SPK として書かれている場合も、書かれていない場合も拾う
# 例:
#   # [SIG RMS] model: ... RMS=..., RMS_std=...
#   # [SPK RMS] model: ... RMS=..., RMS_std=...   (←SIG側もこうなるログがある)
re_rms_any = re.compile(
    rf"\[(SIG|SPK)\s+RMS\]\s+(model|pers|clim)\s*:.*?"
    rf"RMS=({NUM}),\s*RMS_std=({NUM})"
)

kpc = {}
rms = {}

files = sorted(glob.glob(LOG_GLOB))
if not files:
    raise SystemExit(f"No log files matched: {LOG_GLOB}")

for path in files:
    mlead = re_lead.search(path)
    if not mlead:
        continue
    lead = int(mlead.group(1))

    # unlabeled救済用：target別に何回出たか
    order = {"model": 0, "pers": 0, "clim": 0}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # ---- kPC ----
            mk = re_kpc.search(line)
            if mk:
                kernel = mk.group(1)
                target = mk.group(2)
                mean = float(mk.group(3))
                std  = float(mk.group(5))
                kpc[(kernel, lead, target)] = (mean, std)
                continue

            # ---- RMS ----
            mr = re_rms_any.search(line)
            if mr:
                kernel_raw = mr.group(1)      # "SIG" or "SPK"（ログ表記）
                target = mr.group(2)          # model/pers/clim
                mean = float(mr.group(3))
                std  = float(mr.group(4))

                # 重要：ログが常に [SPK RMS] になってしまう場合に備えて振り分ける
                if ASSUME_RMS_1ST_SPK_2ND_SIG and (kernel_raw == "SPK"):
                    if order[target] == 0:
                        kernel = "SPK"
                    elif order[target] == 1:
                        kernel = "SIG"
                    else:
                        print(f"[WARN] lead={lead} target={target}: RMS appears >=3 times in {path}. Ignored.")
                        order[target] += 1
                        continue
                    order[target] += 1
                else:
                    # ちゃんと [SIG RMS] が出るログならそのまま採用
                    kernel = kernel_raw

                key = (kernel, lead, target)
                if key in rms:
                    # 同じキーが複数回出たら上書き事故なので警告（必要なら平均に変えてもよい）
                    print(f"[WARN] duplicate RMS key {key} in {path}. Overwriting.")
                rms[key] = (mean, std)
                continue

def get_series(dic, kernel, target, leads):
    y, e = [], []
    for s in leads:
        key = (kernel, s, target)
        if key in dic:
            yy, ee = dic[key]
        else:
            yy, ee = np.nan, np.nan
        y.append(yy)
        e.append(ee)
    return np.array(y, float), np.array(e, float)

leads = sorted({k[1] for k in kpc.keys()} | {k[1] for k in rms.keys()})
leads = np.array(leads, int)

# -------- kPC --------
MARKER = {"SIG": "o", "SPK": "^"}
plt.figure(figsize=(6.5, 4.2))

# model: SIG, SPK を描く
for kernel in ["SIG", "SPK"]:
    y, e = get_series(kpc, kernel, "model", leads)
    if not np.all(np.isnan(y)):
        plt.errorbar(leads, y, yerr=e, marker=MARKER[kernel], linestyle="-",
                     capsize=3, label=f"{kernel}")

# clim: SIG だけ描いて凡例は "Clim"
y, e = get_series(kpc, "SIG", "clim", leads)
if not np.all(np.isnan(y)):
    plt.errorbar(leads, y, yerr=e, marker="s", linestyle="--",
                 capsize=3, label="Clim")

plt.xticks(leads)                 # leads = [1,2,3,4,5] を想定
plt.xlim(leads.min()-0.1, leads.max()+0.1)  # 任意（端を少し余白）    
plt.xlabel("Lead time s (years)")
plt.ylabel("mean kPC")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_KPC, dpi=300)
plt.close()

# -------- RMS --------
plt.figure(figsize=(6.5, 4.2))

# model: SIG, SPK を描く
for kernel in ["SIG", "SPK"]:
    y, e = get_series(rms, kernel, "model", leads)
    if not np.all(np.isnan(y)):
        plt.errorbar(leads, y, yerr=e, marker=MARKER[kernel], linestyle="-",
                     capsize=3, label=f"{kernel}")

# clim: SIG だけ描いて凡例は "Clim"
y, e = get_series(rms, "SIG", "clim", leads)
if not np.all(np.isnan(y)):
    plt.errorbar(leads, y, yerr=e, marker="s", linestyle="--",
                 capsize=3, label="Clim")

# pers: いずれも描かない（SPK pers, SIG pers とも除外）

plt.xticks(leads)                 # leads = [1,2,3,4,5] を想定
plt.xlim(leads.min()-0.1, leads.max()+0.1)  # 任意（端を少し余白）
plt.xlabel("Lead time s (years)")
plt.ylabel("RMS")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_RMS, dpi=300)
plt.close()

print(f"Saved: {OUT_KPC}")
print(f"Saved: {OUT_RMS}")

# 追加：SIG model が拾えているか最低限の確認
for s in leads:
    if ("SIG", int(s), "model") not in rms:
        print(f"[CHECK] missing SIG RMS model at lead={s}")
        
