#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from scipy import signal
def fit_and_corr_trim(y, re, im, years, trim_years=20.0):
    base = np.isfinite(y) & np.isfinite(re) & np.isfinite(im) & np.isfinite(years)
    if base.sum() < 3:
        yhat_full = np.full_like(y, np.nan, dtype=float)
        return np.nan, np.nan, np.nan, np.nan, np.nan, yhat_full, 0

    yrs = years[base]
    y0  = float(np.min(yrs))
    y1  = float(np.max(yrs))

    mask = base & (years >= y0 + trim_years) & (years <= y1 - trim_years)
    yy, rr, ii = y[mask], re[mask], im[mask]
    X = np.column_stack([rr, ii, np.ones_like(rr)])
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    yhat = X @ coef
    r = float(np.corrcoef(yy, yhat)[0, 1]) if len(yy) >= 2 else np.nan
    a, b, c = map(float, coef)
    theta_deg = float(np.degrees(np.arctan2(b, a)))
    yhat_full = np.full_like(y, np.nan, dtype=float)
    yhat_full[mask] = yhat
    return r, a, b, c, theta_deg, yhat_full, int(mask.sum())
def fit_and_corr(y, re, im):
    """Fit y ≈ a*re + b*im + c and return (r, a, b, c, theta_deg, yhat)."""
    mask = np.isfinite(y) & np.isfinite(re) & np.isfinite(im)
    yy, rr, ii = y[mask], re[mask], im[mask]
    X = np.column_stack([rr, ii, np.ones_like(rr)])
    coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
    yhat = X @ coef
    r = float(np.corrcoef(yy, yhat)[0, 1]) if len(yy) >= 2 else np.nan
    a, b, c = map(float, coef)
    theta_deg = float(np.degrees(np.arctan2(b, a)))
    # yhat_full aligned to original length
    yhat_full = np.full_like(y, np.nan, dtype=float)
    yhat_full[mask] = yhat
    return r, a, b, c, theta_deg, yhat_full

def bandpass_years(y, p_low=20.0, p_high=35.0, fs=1.0, order=4):
    """Bandpass with periods [p_low, p_high] years (p_low < p_high)."""
    out = np.full_like(y, np.nan, dtype=float)
    mask = np.isfinite(y)
    yy = y[mask]
    if yy.size < 3:
        return out

    f_low  = 1.0 / p_high   # cycles/yr
    f_high = 1.0 / p_low
    nyq = fs / 2.0
    Wn = [f_low/nyq, f_high/nyq]
    sos = signal.butter(order, Wn, btype="bandpass", output="sos")
    out[mask] = signal.sosfiltfilt(sos, yy)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpi_txt", default="tpi_from_true_sst_ersstv5.txt",
                    help="monthly TPI text: time_decimal  tpi")
    ap.add_argument("--mode_csv", default="frames/timeseries_155.csv",
                    help="mode timeseries csv")
    ap.add_argument("--use", choices=["raw", "proc"], default="raw",
                    help="use raw or processed (deamp/rot) columns from mode_csv")
    ap.add_argument("--L", type=int, default=12,
                    help="annual window length in months (Aug->Jul is 12)")
    ap.add_argument("--bandpass", action="store_true",
                    help="apply 20–35 year bandpass to annual TPI")
    ap.add_argument("--p_low", type=float, default=20.0, help="bandpass low period (years)")
    ap.add_argument("--p_high", type=float, default=35.0, help="bandpass high period (years)")
    ap.add_argument("--out_csv", default="tpi_and_modefit.csv",
                    help="output CSV filename")
    ap.add_argument("--out_txt", default="tpi_and_modefit.txt",
                    help="output gnuplot text filename (space separated)")
    ap.add_argument("--write_txt", action="store_true",
                    help="also write space-separated txt for gnuplot")
    args = ap.parse_args()

    # ---- load monthly TPI ----
    tpi_data = np.loadtxt(args.tpi_txt, comments="#")
    tpi_time = tpi_data[:, 0]
    tpi_val  = tpi_data[:, 1]

    # ---- load mode ----
    dfm = pd.read_csv(args.mode_csv)
    years = dfm["year"].to_numpy()

    if args.use == "raw":
        re = dfm["real_raw"].to_numpy() if "real_raw" in dfm.columns else dfm["real"].to_numpy()
        im = dfm["imag_raw"].to_numpy() if "imag_raw" in dfm.columns else dfm["imag"].to_numpy()
    else:
        re = dfm["real_proc"].to_numpy()
        im = dfm["imag_proc"].to_numpy()

    # ---- align annual TPI to mode years (definition-fixed; no shift sweep) ----
    # t0 is first monthly timestamp (expected to be mid-month like 1854.625 for Aug 1854)
    t0 = float(tpi_time[0])
    i0 = np.rint(12.0 * (years - t0)).astype(int) 

    print("[debug] t0 =", t0, " (should be 1854.625 for Aug 1854)")
    print("[debug] years[:3] =", years[:3], " i0[:3] =", i0[:3], " tpi_time[i0[:3]] =", tpi_time[i0[:3]])
    
    L = int(args.L)
    tpi_ann = np.full_like(years, np.nan, dtype=float)
    for k, i in enumerate(i0):
        if 0 <= i and i + L <= len(tpi_val):
            tpi_ann[k] = float(np.mean(tpi_val[i:i+L]))

    # ---- correlations on ANN ----
#    r_ann, a1, b1, c1, th1, u_ann = fit_and_corr(tpi_ann, re, im)
#    print(f"[ANN] corr={r_ann:.6f}  a={a1:.6g} b={b1:.6g} c={c1:.6g}  theta(deg)={th1:.2f}")
    r_ann, a1, b1, c1, th1, u_ann, nuse_ann = fit_and_corr_trim(tpi_ann, re, im, years, trim_years=20.0)
    print(f"[ANN trim] corr={r_ann:.6f}  n={nuse_ann}  theta(deg)={th1:.2f}")
    # ---- optional bandpass and correlations ----
    if args.bandpass:
        tpi_bp = bandpass_years(tpi_ann, p_low=args.p_low, p_high=args.p_high)
        r_bp, a2, b2, c2, th2, u_bp, nuse = fit_and_corr_trim(tpi_bp, re, im, years, trim_years=20.0)
        print(f"[BP ] corr={r_bp:.6f}  n={nuse}  a={a2:.6g} b={b2:.6g} c={c2:.6g}  theta(deg)={th2:.2f}")
#        r_bp, a2, b2, c2, th2, u_bp = fit_and_corr(tpi_bp, re, im)
#print(f"[BP ] corr={r_bp :.6f}  a={a2:.6g} b={b2:.6g} c={c2:.6g}  theta(deg)={th2:.2f}")
    else:
        tpi_bp = np.full_like(tpi_ann, np.nan, dtype=float)
        u_bp   = np.full_like(tpi_ann, np.nan, dtype=float)

    # ---- output combined CSV ----
    df_out = pd.DataFrame({
        "year": years,
        "tpi_ann_aug_jul": tpi_ann,
        "tpi_bp": tpi_bp,              # bandpass applied iff --bandpass
        "mode_fit_ann": u_ann,          # best linear combo for ANN
        "mode_fit_bp":  u_bp,           # best linear combo for BP (iff --bandpass)
    })
    df_out.to_csv(args.out_csv, index=False, float_format="%.6f")
    print("wrote:", args.out_csv)

    # ---- optional gnuplot-friendly txt ----
    if args.write_txt:
        with open(args.out_txt, "w", encoding="utf-8") as f:
            f.write("# year  tpi_ann  tpi_bp  mode_fit_ann  mode_fit_bp\n")
            for row in df_out.itertuples(index=False):
                f.write(f"{row.year:10.6f}  {row.tpi_ann_aug_jul: .8f}  {row.tpi_bp: .8f}  "
                        f"{row.mode_fit_ann: .8e}  {row.mode_fit_bp: .8e}\n")
        print("wrote:", args.out_txt)

if __name__ == "__main__":
    main()
    
