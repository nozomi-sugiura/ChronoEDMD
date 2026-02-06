#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

def grab(txt: str, pattern: str) -> float:
    mo = re.search(pattern, txt)
    return float(mo.group(1)) if mo else np.nan

ap = argparse.ArgumentParser()
ap.add_argument("--cv_mode", type=str, default="lfo", choices=["lfo", "lso"],
                help="cv mode directory prefix (lfo or lso)")
ap.add_argument("--s", type=int, default=5,
                help="lead time s used in directory name like lfo_08_s05")
ap.add_argument("--c_area", type=float, default=1.0,
                help="area-mean RMS correction factor: RMS_area = c_area * RMS_now")
args = ap.parse_args()

cv_mode = args.cv_mode
s = int(args.s)
s2 = f"{s:02d}"
c_area = float(args.c_area)

# ---- directory discovery ----
# New style: lfo_01_s05 .. lfo_12_s05
dirs_new = sorted(glob.glob(f"{cv_mode}_[0-1][0-9]_s{s2}"))
# Backward compat: lfo_01 .. lfo_12
dirs_old = sorted(glob.glob(f"{cv_mode}_[0-1][0-9]"))

# Prefer new-style if exists; otherwise fall back to old-style
dirs = dirs_new if len(dirs_new) > 0 else dirs_old

if not dirs:
    raise SystemExit(f"No directories found for cv_mode={cv_mode}. "
                     f"Tried: {cv_mode}_[0-1][0-9]_s{s2} and {cv_mode}_[0-1][0-9]")

print(f"# cv_mode={cv_mode}  s={s}  ndirs={len(dirs)}")
rows = []

for d in dirs:
    print(d)

    # Parse start month from directory name
    # - new: lfo_08_s05 -> 8
    # - old: lfo_08 -> 8
    parts = d.split("_")
    m = int(parts[1])  # "08" -> 8

    logs = sorted(glob.glob(os.path.join(d, "loo_koopman_eval_*_lmda*.log")))
    if not logs:
        continue
    path = logs[0]
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()

    # --- RMS ---
    rms_models = re.findall(r"\[SPK RMS\]\s*model:\s*MSE=.*?,\s*RMS=([0-9.eE+-]+)", txt)
    rms_spk_model = float(rms_models[0]) if len(rms_models) >= 1 else np.nan
    rms_sig_model = float(rms_models[1]) if len(rms_models) >= 2 else np.nan

    # --- RMS clim (ログでは [SPK RMS] clim が2回出る想定) ---
    rms_clims = re.findall(r"\[SPK RMS\]\s*clim\s*:\s*MSE=.*?,\s*RMS=([0-9.eE+-]+)", txt)
    rms_spk_clim = float(rms_clims[0]) if len(rms_clims) >= 1 else np.nan
    rms_sig_clim = float(rms_clims[1]) if len(rms_clims) >= 2 else np.nan

    # --- kPC ---
    # 旧ログ表現の揺れに備えて SIG のパターンを2つ試す
    kpc_sig_model = grab(txt, r"\[SIG kPC\]\s*mean_kPC_model\s*=\s*([0-9.eE+-]+)")
    if np.isnan(kpc_sig_model):
        kpc_sig_model = grab(txt, r"\[SIG kPC\]\s*mean_kPC_model'\s*=\s*([0-9.eE+-]+)")
    kpc_spk_model = grab(txt, r"\[SPK kPC\]\s*mean_kPC_model\s*=\s*([0-9.eE+-]+)")
    kpc_sig_clim  = grab(txt, r"\[SIG kPC\]\s*mean_kPC_clim\s*=\s*([0-9.eE+-]+)")
    kpc_spk_clim  = grab(txt, r"\[SPK kPC\]\s*mean_kPC_clim\s*=\s*([0-9.eE+-]+)")

    rows.append((m,
                 rms_sig_model, rms_spk_model, rms_sig_clim,
                 kpc_sig_model, kpc_spk_model, kpc_sig_clim, kpc_spk_clim))

rows.sort(key=lambda x: x[0])
m = np.array([r[0] for r in rows], dtype=int)

rms_sig_model = np.array([r[1] for r in rows], dtype=float)
rms_spk_model = np.array([r[2] for r in rows], dtype=float)
rms_sig_clim  = np.array([r[3] for r in rows], dtype=float)

# ---- apply area-mean correction (scalar factor) ----
rms_sig_model *= c_area
rms_spk_model *= c_area
rms_sig_clim  *= c_area

kpc_sig_model = np.array([r[4] for r in rows], dtype=float)
kpc_spk_model = np.array([r[5] for r in rows], dtype=float)
kpc_sig_clim  = np.array([r[6] for r in rows], dtype=float)
kpc_spk_clim  = np.array([r[7] for r in rows], dtype=float)

# ----------------------
# Figure 1: RMS
# ----------------------
fig1, ax = plt.subplots(figsize=(10.5, 4.6))
ax.plot(m, rms_sig_model, marker="o", linestyle="-",  label="RMS (SIG model)")
ax.plot(m, rms_spk_model, marker="x", linestyle="-",  label="RMS (SPK model)")
ax.plot(m, rms_sig_clim,  marker="o", linestyle="--", label="RMS (clim)")

ax.set_xlabel("Start month m")
ax.set_ylabel("RMS (area-mean)")
ax.set_xticks(m)
ax.grid(True, alpha=0.25)
ax.set_title(f"Start-month comparison ({cv_mode}, s={s}): RMS (SIG/SPK vs climatology)")

i_min = int(np.nanargmin(rms_sig_model))
ax.annotate(f"min RMS(SIG) @ m={m[i_min]}",
            xy=(m[i_min], rms_sig_model[i_min]),
            xytext=(m[i_min]-1.3, rms_sig_model[i_min]+0.008),
            arrowprops=dict(arrowstyle="->"))

ax.legend(loc="center", framealpha=0)
fig1.tight_layout()
fig1.savefig(f"startmonth_RMS_{cv_mode}_s{s2}_SIG_SPK_vs_clim.png", dpi=200)
fig1.savefig(f"startmonth_RMS_{cv_mode}_s{s2}_SIG_SPK_vs_clim.pdf")
plt.close(fig1)

# ----------------------
# Figure 2: kPC
# ----------------------
fig2, ax = plt.subplots(figsize=(10.5, 4.6))
ax.plot(m, kpc_sig_model, marker="s", linestyle="--", label="mean kPC (SIG model)")
ax.plot(m, kpc_spk_model, marker="d", linestyle="-",  label="mean kPC (SPK model)")
ax.plot(m, kpc_sig_clim,  marker="^", linestyle=":",  label="mean kPC (clim)")

ax.set_xlabel("Start month m")
ax.set_ylabel("mean kPC")
ax.set_xticks(m)
ax.grid(True, alpha=0.25)
ax.set_title(f"Start-month comparison ({cv_mode}, s={s}): mean kPC (SIG/SPK vs climatology)")

i_max = int(np.nanargmax(kpc_sig_model))
ax.annotate(f"max kPC(SIG) @ m={m[i_max]}",
            xy=(m[i_max], kpc_sig_model[i_max]),
            xytext=(m[i_max]+0.3, kpc_sig_model[i_max]-0.018),
            arrowprops=dict(arrowstyle="->"))

ax.legend(loc="center", framealpha=0)
fig2.tight_layout()
fig2.savefig(f"startmonth_kPC_{cv_mode}_s{s2}_SIG_SPK_vs_clim.png", dpi=200)
fig2.savefig(f"startmonth_kPC_{cv_mode}_s{s2}_SIG_SPK_vs_clim.pdf")
plt.close(fig2)

print("Saved:")
print(f"  startmonth_RMS_{cv_mode}_s{s2}_SIG_SPK_vs_clim.(png/pdf)")
print(f"  startmonth_kPC_{cv_mode}_s{s2}_SIG_SPK_vs_clim.(png/pdf)")
