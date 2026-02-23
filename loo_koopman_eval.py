#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LOO Koopman evaluation for SIG / SPK kernels (single-file script).

Policy:
  • SIG: choose (NUM_REJ, LMDA) by maximizing mean kPC (model-vs-truth kPC)
  • SPK: choose NUM_REJ by maximizing mean kPC (model-vs-truth kPC)
  • RMS/clim are reference diagnostics only
  • Additionally: compute climatology-vs-truth kPC (kPC_clim) for the BEST parameters (no optimization uses it)

CV modes:
  • lfo (default): Leave-Future-Out, train with past transitions up to (tau-1)->tau (ALLOWED).
  • lso          : Leave-s-out, remove transitions whose start index i is in
                   [tau-gap, tau+s+gap] (inclusive), and train on all other transitions.
                   Hence (tau-1)->tau is allowed, and (tau+s+1)->(tau+s+2) is allowed when gap=0.

kPC definition (per tau):
  kPC(tau) = K(T_tau, P_tau) / sqrt(K(T_tau,T_tau) K(P_tau,P_tau)),
  where T_tau, P_tau are cumulative-sum paths (prepend 0) built from 12 monthly increments.

Inputs required in cwd:
  - sst_compressed.npy : (n_months, d) anomaly in degC on valid gridpoints
  - clim_compressed.npy: (n_months, d) climatology in degC on same valid gridpoints
  - valid_indices.npy  : indices into (ny*nx) flattened grid
  - Mw_orders.npz      : contains M_order0..M_orderORD for SIG
  - sigma.npy          : scalar sigma used for signature base kernel (kPC diagnostic kernel)
"""

import os
import sys
import time
import traceback
import argparse
from functools import partial
from datetime import datetime, timezone, timedelta

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import gridspec


# =========================
# kernels for signature-kPC (diagnostic / objective only)
# =========================
def rbf_kernel(x, y, sigma):
    return np.exp(-np.sum((x - y) ** 2) / (2.0 * sigma ** 2))

def poly_kernel(x, y, sigma, c=0.0, d=1):
    return (np.dot(x, y) / sigma ** 2 + c) ** d

def base_kernel_factory(kernel_type="rbf", sigma=None, c=0.0, d=1):
    if kernel_type == "rbf":
        return partial(rbf_kernel, sigma=float(sigma))
    if kernel_type == "poly":
        return partial(poly_kernel, sigma=float(sigma), c=float(c), d=int(d))
    raise ValueError(f"Unknown kernel type: {kernel_type}")


# =========================
# signature kernel (discrete DP)
# =========================
def suffix_sum_strict(A):
    # inclusive suffix sum, then strict shift
    S = np.cumsum(np.cumsum(A[::-1, ::-1], axis=0), axis=1)[::-1, ::-1]
    out = np.zeros_like(A)
    out[:-1, :-1] = S[1:, 1:]
    return out

def signature_kernel(X, Y, m, base_kernel):
    nx = X.shape[0]
    ny = Y.shape[0]

    kij = np.zeros((nx, ny), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            kij[i, j] = base_kernel(X[i], Y[j])

    # discrete second-difference kernel on the grid
    K = kij[1:, 1:] + kij[:-1, :-1] - kij[:-1, 1:] - kij[1:, :-1]

    A = K.copy()
    cum_prev = A.sum()
    level_list = [1.0, cum_prev]  # level 0 is 1

    for d in range(2, m + 1):
        Q = suffix_sum_strict(A)
        A = K * (1.0 + Q)
        cum = A.sum()
        level_list.append(cum - cum_prev)  # exactly degree d
        cum_prev = cum

    total = 1.0 + cum_prev
    return level_list, total

def cumulative_path_from_monthly(monthly, include_zero=True):
    monthly = np.asarray(monthly, dtype=np.float64)
    c = np.cumsum(monthly, axis=0)
    if include_zero:
        out = np.zeros((c.shape[0] + 1, c.shape[1]), dtype=np.float64)  # zero-padding
        out[1:, :] = c
        return out
    return c

def signature_kernel_pc_diag(t_monthly, p_monthly, m, base_kernel, lmda=1.0,
                             include_zero=True, eps=1e-12):
    T = cumulative_path_from_monthly(t_monthly, include_zero=include_zero)
    P = cumulative_path_from_monthly(p_monthly, include_zero=include_zero)

    lev_tp, _ = signature_kernel(T, P, m, base_kernel)
    lev_tt, _ = signature_kernel(T, T, m, base_kernel)
    lev_pp, _ = signature_kernel(P, P, m, base_kernel)

    lev_tp = np.asarray(lev_tp, dtype=np.float64)
    lev_tt = np.asarray(lev_tt, dtype=np.float64)
    lev_pp = np.asarray(lev_pp, dtype=np.float64)

    w = np.array([lmda ** (2 * l) for l in range(m + 1)], dtype=np.float64)

    ktp = float(np.dot(w, lev_tp))
    ktt = float(np.dot(w, lev_tt))
    kpp = float(np.dot(w, lev_pp))

    den = np.sqrt(max(ktt, 0.0) * max(kpp, 0.0))
    kpc = np.nan if den < eps else float(ktp / den)

    info = {
        "weights": w,
        "TP_levels": lev_tp, "TT_levels": lev_tt, "PP_levels": lev_pp,
        "TP_wlevels": w * lev_tp, "TT_wlevels": w * lev_tt, "PP_wlevels": w * lev_pp,
        "KTP": ktp, "KTT": ktt, "KPP": kpp
    }
    return kpc, info


# =========================
# area weights + plotting
# =========================
def compute_flat_area_weights(valid_indices, ny=89, nx=180, normalize=False):
    latitudes = np.linspace(-88, 88, ny)
    coslat = np.cos(np.deg2rad(latitudes))
    weights_2d = np.repeat(coslat[:, np.newaxis], nx, axis=1)
    weights_flat = weights_2d.reshape(-1)
    w = np.sqrt(weights_flat[valid_indices])
    if normalize:
        w = w / np.sqrt(np.sum(w ** 2))
    return w

def plot_map_pdf(field2d, fname, title="", vmin=None, vmax=None, cmap="viridis", dpi=300,
                 cbar_label="", bad_color="#654321",
                 lon0=0.0, lon1=360.0, lat0=-90.0, lat1=90.0,
                 fig_w=10.0,
                 wspace=0.08, right=0.90):

    A = np.asarray(field2d)
    ny, nx = A.shape

    extent = (lon0, lon1, lat0, lat1)

    lon_span = float(lon1 - lon0)
    lat_span = float(lat1 - lat0)
    fig_h = fig_w * (lat_span / lon_span)

    xticks = np.linspace(lon0, lon1, 7)
    yticks = np.linspace(lat0, lat1, 5)

    cmap_obj = plt.get_cmap(cmap)
    cmap_obj = cmap_obj.copy() if hasattr(cmap_obj, "copy") else cmap_obj
    if hasattr(cmap_obj, "set_bad"):
        cmap_obj.set_bad(color=bad_color)

    if vmin is None and vmax is None:
        vmax0 = float(np.nanmax(np.abs(A)))
        vmin, vmax = -vmax0, vmax0
    elif vmin is None:
        vmin = float(np.nanmin(A))
    elif vmax is None:
        vmax = float(np.nanmax(A))

    fig = plt.figure(figsize=(fig_w, fig_h))
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 0.04])

    ax = fig.add_subplot(spec[0])
    cax = fig.add_subplot(spec[1])

    im = ax.imshow(A, origin="lower", cmap=cmap_obj, vmin=vmin, vmax=vmax,
                   extent=extent, aspect="equal")

    ax.set_title(title)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    cbar = fig.colorbar(im, cax=cax)
    if cbar_label:
        cbar.set_label(cbar_label)

    fig.subplots_adjust(wspace=wspace, right=right)

    outdir = os.path.dirname(fname) or "."
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(fname, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"# [SAVED] {os.path.abspath(fname)}")


# =========================
# Koopman helpers
# =========================
def sorted_pair_indices(res, eigvals, tol=1e-8):
    n = len(eigvals)
    used = np.zeros(n, dtype=bool)
    groups = []
    for i in range(n):
        if used[i]:
            continue
        vi = eigvals[i]
        pair = [i]
        for j in range(i + 1, n):
            if used[j]:
                continue
            vj = eigvals[j]
            if np.abs(vi - np.conj(vj)) < tol:
                pair.append(j)
                used[j] = True
                break
        used[i] = True
        groups.append(pair)

    group_res = [(g, sum(res[ii] for ii in g)) for g in groups]
    group_res.sort(key=lambda x: x[1])
    return [g for g, _ in group_res]

def residual2(K2, L, Tl):
    res = np.zeros(L.shape[0])
    for j in range(L.shape[0]):
        res0 = Tl[:, j].T.conj() @ Tl[:, j]
        res1 = Tl[:, j].T.conj() @ K2 @ Tl[:, j]
        res[j] = np.sqrt(max(0.0, (res1.real / res0.real) - np.abs(L[j]) ** 2))
    return res

def build_train_indices(tau: int, M: int, s: int, cv_mode: str = "lfo", gap: int = 0):
    """
    Returns (idx_next, idx_state) for training using 1-step transitions i -> i+1.

    lfo:
      train on past transitions up to (tau-1)->tau (allowed).
      i ranges 0..tau-1-gap. With gap=0, includes (tau-1)->tau.

    lso:
      exclude transitions whose start index i lies in
        [tau-gap, tau+s+gap] (inclusive),
      and use all other transitions.
      With gap=0, this excludes i=tau, tau+1, ..., tau+s.
      Hence (tau-1)->tau is allowed, and (tau+s+1)->(tau+s+2) is allowed.
    """
    if M < 2:
        return [], []
    if gap < 0:
        raise ValueError("gap must be nonnegative")

    if cv_mode == "lfo":
        t_obs = tau - gap
        if t_obs < 1:
            return [], []
        idx_state = list(range(0, t_obs))      # 0..tau-1-gap
        idx_next  = [i + 1 for i in idx_state]
        return idx_next, idx_state

    if cv_mode == "lso":
        i_all = np.arange(0, M - 1, dtype=int)  # transition starts: 0..M-2
        lo = tau - gap
        hi = tau + s + gap                      # inclusive
        mask = (i_all < lo) | (i_all > hi)
        idx_state = i_all[mask].tolist()
        idx_next  = (i_all[mask] + 1).tolist()
        return idx_next, idx_state

    raise ValueError(f"Unknown cv_mode: {cv_mode}")


# =========================
# SPK
# =========================
def sigma_spk_numpy(X: np.ndarray, I: int = 12) -> float:
    """
    SPK 用: 同じ月 i の中で年 j1,j2 のみ比較する（全ての (i, j1, j2)）
      各 i について sigma_i^2 = E||X_i(j1) - X_i(j2)||^2
                    = 2(E||X_i||^2 - ||E X_i||^2)
      最後に i で平均：sigma^2 = mean_i sigma_i^2
    X: (n, d, ntr)
    """
    X = np.asarray(X, dtype=np.float64)
    I = int(I)
    X = X[:I, :, :]                 # (I, d, ntr)

    mean_norm2 = np.mean(np.sum(X * X, axis=1), axis=1)   # (I,)
    mu = np.mean(X, axis=2)                               # (I, d)
    mu_norm2 = np.sum(mu * mu, axis=1)                    # (I,)

    sigma2_i = 2.0 * (mean_norm2 - mu_norm2)              # (I,)
    sigma2 = float(np.mean(sigma2_i))
    return float(np.sqrt(max(sigma2, 0.0)))

def spk_month_weights(kind="cumsum"):
    w = np.ones(13, dtype=np.float64)
    if kind == "cumsum":
        w[0] = 1.0
        w[12] = 0.0
    elif kind == "nocumsum":
        w[0] = 0.5
        w[12] = 0.5
    else:
        raise ValueError(f"Unknown spk_weight kind: {kind}")
    return w

def sum_of_pairs_gram_from_windows(X0, sigma, month_w=None, normalize_diag=False):
    n, d, ntr = X0.shape
    assert n == 13
    if month_w is None:
        month_w = np.ones(13, dtype=np.float64)
        month_w[0] = 1.0
        month_w[12] = 0.0
    month_w = np.asarray(month_w, dtype=np.float64)

    def k_rbf(x, y):
        return float(np.exp(-np.sum((x - y) ** 2) / (2.0 * sigma ** 2)))

    H = np.zeros((ntr, ntr), dtype=np.float64)
    for i in range(ntr):
        Xi = X0[:, :, i]
        for j in range(i + 1):
            Xj = X0[:, :, j]
            ssum = 0.0
            for m in range(13):
                ssum += month_w[m] * k_rbf(Xi[m], Xj[m])
            H[i, j] = ssum
            H[j, i] = ssum

    if normalize_diag:
        diag = np.sqrt(np.maximum(np.diag(H), 1e-30))
        H = H / diag[:, None] / diag[None, :]
    return H

def build_windows_tensor_from_monthly(sst_compressed, wt, n=13, stride=12, start_offset=0):
    n_months, d = sst_compressed.shape
    n_windows = (n_months - (start_offset + n)) // stride + 1
    X0 = np.zeros((n, d, n_windows), dtype=np.float64)
    for i in range(n_windows):
        start = start_offset + i * stride
        end = start + n
        X0[:, :, i] = sst_compressed[start:end, :] * wt[None, :]
    return X0


# =========================
# Xfeat (concat12 is required for kPC objective)
# =========================
def build_Xfeat_concat12(sst_compressed, wt, stride=12, start_offset=0):
    n_months, d = sst_compressed.shape
    M = (n_months - (start_offset + 13)) // stride + 1
    Xcat = np.zeros((M, 12 * d), dtype=np.float64)
    for i in range(M):
        start = start_offset + i * stride
        window = sst_compressed[start:start + 13, :] * wt[None, :]
        months12 = window[:12, :]
        Xcat[i, :] = months12.reshape(-1)
    return Xcat[None, :, :]  # (1, M, 12*d)

def build_Xfeat(sst_compressed, wt, mode="concat12", stride=12, start_offset=0):
    if mode != "concat12":
        raise ValueError("This script is intended for --xfeat concat12.")
    return build_Xfeat_concat12(sst_compressed, wt, stride=stride, start_offset=start_offset)


# =========================
# Climatology prediction in concat12 feature space
# =========================
def y_clim_concat12_from_clim(
    clim_compressed: np.ndarray,
    wt: np.ndarray,
    tau: int,
    s: int,
    start_offset: int = 0,
    stride: int = 12,
) -> np.ndarray:
    """
    Build y_clim in the same space as Xfeat=concat12 built from anomaly SST.

    anomaly a_t = x_t - c_t  (sst_compressed.npy is anomaly, clim_compressed.npy is c_t)
    Climatology forecast in anomaly coordinates:
        a_hat_{t|tau} = c_{tau_m} - c_t,
    where tau_m is the latest index <= t_obs_last with the same month as t.
    """
    clim_compressed = np.asarray(clim_compressed, dtype=np.float64)
    wt = np.asarray(wt, dtype=np.float64)

    T, d = clim_compressed.shape
    if wt.shape != (d,):
        raise ValueError(f"wt shape {wt.shape} must be (d,) with d={d}.")

    tau = int(tau)
    s = int(s)
    if tau < 0 or s < 0:
        raise ValueError("tau and s must be nonnegative integers.")
    if not (0 <= start_offset <= 11):
        raise ValueError("start_offset must be in 0..11.")

    t_obs_last = start_offset + stride * tau + (stride - 1)
    if t_obs_last < 0 or t_obs_last >= T:
        raise ValueError(f"t_obs_last={t_obs_last} out of range for T={T}.")

    t0 = start_offset + stride * (tau + s)
    if t0 + 11 >= T:
        raise ValueError(f"Target window [{t0}..{t0+11}] out of range for T={T}.")

    y = np.empty((12, d), dtype=np.float64)
    for k in range(12):
        t_future = t0 + k
        m = t_future % 12
        tau_m = t_obs_last - ((t_obs_last - m) % 12)
        if tau_m < 0:
            tau_m = m
            if tau_m >= T:
                raise ValueError("Not enough data for same-month climatology.")
        y[k, :] = (clim_compressed[tau_m, :] - clim_compressed[t_future, :]) * wt

    return y.reshape(-1)


# =========================
# RMS criterion (reference only)
# =========================
def monthwise_mean_sqnorm(err_vec, nmon=12):
    L = err_vec.size
    if L % nmon != 0:
        raise ValueError("length not divisible by nmon")
    d = L // nmon
    E = err_vec.reshape(nmon, d)
    return float(np.mean(np.sum(E * E, axis=1)))

def mse_rms_error(params, M0, X_all, s=5, eps=1e-12,
                  xfeat_mode="concat12",
                  min_coverage=0.8,
                  verbose=False,
                  tau_start=1,
                  cv_mode="lfo",
                  gap=0,
                  clim_compressed=None, wt=None, start_offset=0,
                  stride=12,
                  return_std=False):
    """
    If return_std=False:
      (mse_model, rms_model, mse_pers, rms_pers, mse_clim, rms_clim)

    If return_std=True:
      (mse_model, rms_model, std_rms_model,
       mse_pers,  rms_pers,  std_rms_pers,
       mse_clim,  rms_clim,  std_rms_clim)

    NOTE: std is computed over per-tau RMS values, restricted to groups that satisfy min_coverage.
    """
    if clim_compressed is None:
        raise ValueError("mse_rms_error: clim_compressed is None.")
    if wt is None:
        raise ValueError("mse_rms_error: wt is None.")

    LMDA, num_rej = params
    NUM_REJ = int(round(num_rej))

    H = M0[0].copy()
    for i in range(1, len(M0)):
        H += M0[i] * (LMDA ** (2 * i))

    M = H.shape[1]
    tau_total = max(0, (M - s) - int(tau_start))

    D = X_all[0].shape[1]
    if xfeat_mode != "concat12":
        raise ValueError("mse_rms_error expects xfeat_mode='concat12'.")
    if D % 12 != 0:
        raise ValueError("concat12 expects D divisible by 12.")
    d_space = D // 12

    rms_list_model = []
    rms_list_pers  = []
    rms_list_clim  = []

    sum_mse_model = 0.0
    sum_mse_pers  = 0.0
    sum_mse_clim  = 0.0
    n_groups_used = 0

    for X in X_all:
        per_tau_model = []
        per_tau_pers  = []
        per_tau_clim  = []
        used_tau = 0

        tmp_rms_model = []
        tmp_rms_pers  = []
        tmp_rms_clim  = []

        for tau in range(int(tau_start), M - s):
            idx_next, idx_state = build_train_indices(
                tau=tau, M=M, s=s, cv_mode=cv_mode, gap=int(gap)
            )
            if len(idx_state) == 0:
                continue

            G  = H[np.ix_(idx_state, idx_state)]
            A  = H[np.ix_(idx_next,  idx_state)]
            Lm = H[np.ix_(idx_next,  idx_next)]

            S2, Q = np.linalg.eigh(G)
            S2 = np.maximum(S2, 0.0)
            S = np.diag(np.sqrt(S2))
            Sinv = np.linalg.pinv(S)
            K = Sinv @ Q.T @ A @ Q @ Sinv

            eigvals, Vl, V = sp.linalg.eig(K, left=True)
            K2 = Sinv @ Q.T @ Lm @ Q @ Sinv
            res = residual2(K2, eigvals, Vl)
            sorted_groups = sorted_pair_indices(res, eigvals)

            if NUM_REJ == 0:
                groups_to_keep = sorted_groups
            elif NUM_REJ >= len(sorted_groups):
                groups_to_keep = []
            else:
                groups_to_keep = sorted_groups[:len(sorted_groups) - NUM_REJ]

            keep_indices = sorted([ii for g in groups_to_keep for ii in g])
            if len(keep_indices) == 0:
                continue

            eigvals_k = eigvals[keep_indices]
            V_k = V[:, keep_indices]

            k_vec = H[tau, idx_state]
            phi_tau = k_vec @ Q @ Sinv @ V_k
            Phi_x = Q @ S @ V_k

            X_train = X[idx_next]
            Xi = np.linalg.pinv(Phi_x, rcond=eps) @ X_train
            y_pred = np.real((phi_tau * (eigvals_k ** s)) @ Xi)

            y_true = X[tau + s]
            y_pers = X[tau]

            y_clim = y_clim_concat12_from_clim(
                clim_compressed=clim_compressed,
                wt=wt,
                tau=tau, s=s,
                start_offset=int(start_offset),
                stride=int(stride),
            )

            err_model = y_pred - y_true
            err_pers  = y_pers - y_true
            err_clim  = y_clim - y_true

            a_m = monthwise_mean_sqnorm(err_model, nmon=12)  # (1/12)Σ||e||^2
            a_p = monthwise_mean_sqnorm(err_pers,  nmon=12)
            a_c = monthwise_mean_sqnorm(err_clim,  nmon=12)

            per_tau_model.append(a_m)
            per_tau_pers.append(a_p)
            per_tau_clim.append(a_c)

            # per-tau RMS in physical unit:
            # RMS_tau = sqrt( (1/d) * (1/12) Σ ||e||^2 )
            tmp_rms_model.append(float(np.sqrt(a_m / d_space)))
            tmp_rms_pers.append (float(np.sqrt(a_p / d_space)))
            tmp_rms_clim.append (float(np.sqrt(a_c / d_space)))

            used_tau += 1

        coverage = (used_tau / tau_total) if tau_total > 0 else 0.0
        if coverage < float(min_coverage):
            if verbose:
                print(f"# [RMS INVALID] NUM_REJ={NUM_REJ} coverage={coverage:.3f} (used={used_tau}/{tau_total})")
            continue

        # Only add to std lists if this group is valid under min_coverage
        rms_list_model.extend(tmp_rms_model)
        rms_list_pers.extend(tmp_rms_pers)
        rms_list_clim.extend(tmp_rms_clim)

        if per_tau_model:
            sum_mse_model += float(np.mean(per_tau_model)) / d_space
            sum_mse_pers  += float(np.mean(per_tau_pers))  / d_space
            sum_mse_clim  += float(np.mean(per_tau_clim))  / d_space
            n_groups_used += 1

    if n_groups_used == 0:
        if not return_std:
            return (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)
        return (np.inf, np.inf, np.inf,
                np.inf, np.inf, np.inf,
                np.inf, np.inf, np.inf)

    mse_model = sum_mse_model / n_groups_used
    mse_pers  = sum_mse_pers  / n_groups_used
    mse_clim  = sum_mse_clim  / n_groups_used

    rms_model = float(np.sqrt(mse_model))
    rms_pers  = float(np.sqrt(mse_pers))
    rms_clim  = float(np.sqrt(mse_clim))

    if not return_std:
        return (mse_model, rms_model, mse_pers, rms_pers, mse_clim, rms_clim)

    std_rms_model = float(np.std(rms_list_model, ddof=1)) if len(rms_list_model) >= 2 else 0.0
    std_rms_pers  = float(np.std(rms_list_pers,  ddof=1)) if len(rms_list_pers)  >= 2 else 0.0
    std_rms_clim  = float(np.std(rms_list_clim,  ddof=1)) if len(rms_list_clim)  >= 2 else 0.0

    return (mse_model, rms_model, std_rms_model,
            mse_pers,  rms_pers,  std_rms_pers,
            mse_clim,  rms_clim,  std_rms_clim)


# =========================
# Spatial RMS maps (reference only)
# =========================
def loo_spatial_rms_maps(params, M0, X_all, valid_indices, wt,
                         ny=89, nx=180, s=5, eps=1e-12,
                         xfeat_mode="concat12",
                         return_physical=True,
                         tau_start=1,
                         cv_mode="lfo",
                         gap=0,
                         clim_compressed=None, start_offset=0,
                         stride=12):
    if clim_compressed is None:
        raise ValueError("loo_spatial_rms_maps: clim_compressed is None.")
    if wt is None:
        raise ValueError("loo_spatial_rms_maps: wt is None.")

    LMDA, num_rej = params
    NUM_REJ = int(round(num_rej))

    H = M0[0].copy()
    for i in range(1, len(M0)):
        H += M0[i] * (LMDA ** (2 * i))

    M = H.shape[1]
    D = X_all[0].shape[1]
    if D % 12 != 0:
        raise ValueError("concat12 expects D divisible by 12.")
    d = D // 12

    err2_sum_model = np.zeros(d, dtype=np.float64)
    err2_sum_pers  = np.zeros(d, dtype=np.float64)
    err2_sum_clim  = np.zeros(d, dtype=np.float64)
    cnt_sum        = np.zeros(d, dtype=np.int64)

    for X in X_all:
        for tau in range(int(tau_start), M - s):
            idx_next, idx_state = build_train_indices(
                tau=tau, M=M, s=s, cv_mode=cv_mode, gap=int(gap)
            )
            if len(idx_state) == 0:
                continue

            G  = H[np.ix_(idx_state, idx_state)]
            A  = H[np.ix_(idx_next,  idx_state)]
            Lm = H[np.ix_(idx_next,  idx_next)]

            S2, Q = np.linalg.eigh(G)
            S2 = np.maximum(S2, 0.0)
            S = np.diag(np.sqrt(S2))
            Sinv = np.linalg.pinv(S)
            K = Sinv @ Q.T @ A @ Q @ Sinv

            eigvals, Vl, V = sp.linalg.eig(K, left=True)
            K2 = Sinv @ Q.T @ Lm @ Q @ Sinv
            res = residual2(K2, eigvals, Vl)
            sorted_groups = sorted_pair_indices(res, eigvals)

            if NUM_REJ == 0:
                groups_to_keep = sorted_groups
            elif NUM_REJ >= len(sorted_groups):
                continue
            else:
                groups_to_keep = sorted_groups[:len(sorted_groups) - NUM_REJ]

            keep_indices = sorted([ii for g in groups_to_keep for ii in g])
            if len(keep_indices) == 0:
                continue

            eigvals_k = eigvals[keep_indices]
            V_k = V[:, keep_indices]

            k_vec = H[tau, idx_state]
            phi_tau = k_vec @ Q @ Sinv @ V_k
            Phi_x = Q @ S @ V_k

            X_train = X[idx_next]
            Xi = np.linalg.pinv(Phi_x, rcond=eps) @ X_train
            y_pred = np.real((phi_tau * (eigvals_k ** s)) @ Xi)

            y_true = X[tau + s]
            y_pers = X[tau]

            y_clim = y_clim_concat12_from_clim(
                clim_compressed=clim_compressed,
                wt=wt,
                tau=tau, s=s,
                start_offset=int(start_offset),
                stride=int(stride),
            )

            e_model = (y_pred - y_true).reshape(12, d)
            e_pers  = (y_pers - y_true).reshape(12, d)
            e_clim  = (y_clim - y_true).reshape(12, d)

            err2_sum_model += np.mean(e_model ** 2, axis=0)
            err2_sum_pers  += np.mean(e_pers  ** 2, axis=0)
            err2_sum_clim  += np.mean(e_clim  ** 2, axis=0)
            cnt_sum += 1

    ok = cnt_sum > 0
    mse_y_model = np.full(d, np.nan); mse_y_model[ok] = err2_sum_model[ok] / cnt_sum[ok]
    mse_y_pers  = np.full(d, np.nan); mse_y_pers[ok]  = err2_sum_pers[ok]  / cnt_sum[ok]
    mse_y_clim  = np.full(d, np.nan); mse_y_clim[ok]  = err2_sum_clim[ok]  / cnt_sum[ok]

    rms_y_model = np.sqrt(mse_y_model)
    rms_y_pers  = np.sqrt(mse_y_pers)
    rms_y_clim  = np.sqrt(mse_y_clim)

    def to_map(space_vec):
        out = np.full(ny * nx, np.nan, dtype=np.float64)
        out[valid_indices] = space_vec
        return out.reshape(ny, nx)

    rms_y_maps = (to_map(rms_y_model), to_map(rms_y_pers), to_map(rms_y_clim))
    if not return_physical:
        return rms_y_maps

    wt2 = np.maximum(wt ** 2, 1e-12)
    rms_x_model = np.sqrt(mse_y_model / wt2)
    rms_x_pers  = np.sqrt(mse_y_pers  / wt2)
    rms_x_clim  = np.sqrt(mse_y_clim  / wt2)

    rms_x_maps = (to_map(rms_x_model), to_map(rms_x_pers), to_map(rms_x_clim))
    return rms_y_maps, rms_x_maps


# =========================
# Diagnostics helper
# =========================
def summarize_mode_diag(diag):
    def _stat(x):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return "EMPTY"
        return f"mean={x.mean():.2f}, med={np.median(x):.0f}, min={x.min():.0f}, max={x.max():.0f}"
    return (
        f"groups({_stat(diag['n_groups'])}) "
        f"keep_groups({_stat(diag['n_keep_groups'])}) "
        f"keep_pairs({_stat(diag['n_keep_pairs'])}) "
        f"keep_eig({_stat(diag['n_keep_eig'])})"
    )


# =========================
# mean kPC objective (PRIMARY) + mode diagnostics
#   - optimization uses only model kPC
# =========================
def mean_kpc_score(params, M0, X_all, s=5, eps=1e-12,
                   xfeat_mode="concat12",
                   base_kernel=None, ord_sig=7,
                   lmda_kpc=1.0,
                   min_coverage=0.8,
                   max_print=0,
                   verbose=False,
                   tau_start=1,
                   cv_mode="lfo",
                   gap=0,
                   return_diag=False):
    if xfeat_mode != "concat12":
        raise ValueError("mean_kpc_score expects xfeat_mode='concat12'.")
    if base_kernel is None or (not callable(base_kernel)):
        raise TypeError("base_kernel must be callable (from base_kernel_factory).")

    LMDA, num_rej = params
    NUM_REJ = int(round(num_rej))

    H = M0[0].copy()
    for i in range(1, len(M0)):
        H += M0[i] * (LMDA ** (2 * i))

    M = H.shape[1]
    tau_total = max(0, (M - s) - int(tau_start))

    kpc_sum = 0.0
    kpc_cnt = 0
    used_tau = 0
    last_info = None

    diag = {
        "n_groups": [],
        "n_keep_groups": [],
        "n_keep_pairs": [],
        "n_keep_eig": [],
    }

    print_left = int(max_print)

    for X in X_all:
        for tau in range(int(tau_start), M - s):
            idx_next, idx_state = build_train_indices(
                tau=tau, M=M, s=s, cv_mode=cv_mode, gap=int(gap)
            )
            if len(idx_state) == 0:
                continue

            G  = H[np.ix_(idx_state, idx_state)]
            A  = H[np.ix_(idx_next,  idx_state)]
            Lm = H[np.ix_(idx_next,  idx_next)]

            S2, Q = np.linalg.eigh(G)
            S2 = np.maximum(S2, 0.0)
            S = np.diag(np.sqrt(S2))
            Sinv = np.linalg.pinv(S)
            K = Sinv @ Q.T @ A @ Q @ Sinv

            eigvals, Vl, V = sp.linalg.eig(K, left=True)
            K2 = Sinv @ Q.T @ Lm @ Q @ Sinv
            res = residual2(K2, eigvals, Vl)
            sorted_groups = sorted_pair_indices(res, eigvals)

            n_groups = len(sorted_groups)

            if NUM_REJ == 0:
                groups_to_keep = sorted_groups
            elif NUM_REJ >= n_groups:
                groups_to_keep = []
            else:
                groups_to_keep = sorted_groups[: (n_groups - NUM_REJ)]

            keep_indices = sorted([ii for g in groups_to_keep for ii in g])

            n_keep_groups = len(groups_to_keep)
            n_keep_pairs = sum(1 for g in groups_to_keep if len(g) == 2)
            diag["n_groups"].append(n_groups)
            diag["n_keep_groups"].append(n_keep_groups)
            diag["n_keep_pairs"].append(n_keep_pairs)
            diag["n_keep_eig"].append(len(keep_indices))

            if len(keep_indices) == 0:
                continue

            eigvals_k = eigvals[keep_indices]
            V_k = V[:, keep_indices]

            k_vec = H[tau, idx_state]
            phi_tau = k_vec @ Q @ Sinv @ V_k
            Phi_x = Q @ S @ V_k

            X_train = X[idx_next]
            Xi = np.linalg.pinv(Phi_x, rcond=eps) @ X_train
            y_pred = np.real((phi_tau * (eigvals_k ** s)) @ Xi)

            y_true = X[tau + s]

            D = y_true.size
            if D % 12 != 0:
                raise ValueError("concat12 expects D divisible by 12.")
            d_space = D // 12

            t_monthly = y_true.reshape(12, d_space)
            p_monthly = y_pred.reshape(12, d_space)

            kpc, info = signature_kernel_pc_diag(
                t_monthly, p_monthly, m=int(ord_sig),
                base_kernel=base_kernel,
                lmda=float(lmda_kpc),
                include_zero=True
            )

            used_tau += 1

            if np.isfinite(kpc):
                kpc_sum += float(kpc)
                kpc_cnt += 1
                last_info = info

            if verbose and (print_left > 0):
                print(f"# [kPC] tau={tau} kPC={kpc}")
                print_left -= 1

    coverage = (used_tau / tau_total) if tau_total > 0 else 0.0
    if (coverage < float(min_coverage)) or (kpc_cnt == 0):
        if return_diag:
            return (-np.inf, float(coverage), int(kpc_cnt), last_info, diag)
        return (-np.inf, float(coverage), int(kpc_cnt), last_info)

    mean_kpc = kpc_sum / kpc_cnt
    if return_diag:
        return (float(mean_kpc), float(coverage), int(kpc_cnt), last_info, diag)
    return (float(mean_kpc), float(coverage), int(kpc_cnt), last_info)


# =========================
# mean kPC for BEST params: model-kPC and clim-kPC together (with std)
# =========================
def mean_kpc_score_pair_model_clim(params, M0, X_all, s=5, eps=1e-12,
                                  xfeat_mode="concat12",
                                  base_kernel=None, ord_sig=7,
                                  lmda_kpc=1.0,
                                  min_coverage=0.8,
                                  tau_start=1,
                                  cv_mode="lfo",
                                  gap=0,
                                  clim_compressed=None, wt=None,
                                  start_offset=0, stride=12,
                                  return_diag=False):
    if clim_compressed is None or wt is None:
        raise ValueError("mean_kpc_score_pair_model_clim needs clim_compressed and wt.")
    if xfeat_mode != "concat12":
        raise ValueError("mean_kpc_score_pair_model_clim expects xfeat_mode='concat12'.")
    if base_kernel is None or (not callable(base_kernel)):
        raise TypeError("base_kernel must be callable.")

    LMDA, num_rej = params
    NUM_REJ = int(round(num_rej))

    H = M0[0].copy()
    for i in range(1, len(M0)):
        H += M0[i] * (LMDA ** (2 * i))

    M = H.shape[1]
    tau_total = max(0, (M - s) - int(tau_start))

    kpc_list_m = []
    kpc_list_c = []
    used_tau = 0
    last_info = None

    diag = {
        "n_groups": [],
        "n_keep_groups": [],
        "n_keep_pairs": [],
        "n_keep_eig": [],
    }

    for X in X_all:
        for tau in range(int(tau_start), M - s):
            idx_next, idx_state = build_train_indices(
                tau=tau, M=M, s=s, cv_mode=cv_mode, gap=int(gap)
            )
            if len(idx_state) == 0:
                continue

            G  = H[np.ix_(idx_state, idx_state)]
            A  = H[np.ix_(idx_next,  idx_state)]
            Lm = H[np.ix_(idx_next,  idx_next)]

            S2, Q = np.linalg.eigh(G)
            S2 = np.maximum(S2, 0.0)
            S = np.diag(np.sqrt(S2))
            Sinv = np.linalg.pinv(S)
            K = Sinv @ Q.T @ A @ Q @ Sinv

            eigvals, Vl, V = sp.linalg.eig(K, left=True)
            K2 = Sinv @ Q.T @ Lm @ Q @ Sinv
            res = residual2(K2, eigvals, Vl)
            sorted_groups = sorted_pair_indices(res, eigvals)

            n_groups = len(sorted_groups)

            if NUM_REJ == 0:
                groups_to_keep = sorted_groups
            elif NUM_REJ >= n_groups:
                groups_to_keep = []
            else:
                groups_to_keep = sorted_groups[: (n_groups - NUM_REJ)]

            keep_indices = sorted([ii for g in groups_to_keep for ii in g])

            n_keep_groups = len(groups_to_keep)
            n_keep_pairs = sum(1 for g in groups_to_keep if len(g) == 2)
            diag["n_groups"].append(n_groups)
            diag["n_keep_groups"].append(n_keep_groups)
            diag["n_keep_pairs"].append(n_keep_pairs)
            diag["n_keep_eig"].append(len(keep_indices))

            if len(keep_indices) == 0:
                continue

            eigvals_k = eigvals[keep_indices]
            V_k = V[:, keep_indices]

            k_vec = H[tau, idx_state]
            phi_tau = k_vec @ Q @ Sinv @ V_k
            Phi_x = Q @ S @ V_k

            X_train = X[idx_next]
            Xi = np.linalg.pinv(Phi_x, rcond=eps) @ X_train
            y_pred = np.real((phi_tau * (eigvals_k ** s)) @ Xi)

            y_true = X[tau + s]

            D = y_true.size
            if D % 12 != 0:
                raise ValueError("concat12 expects D divisible by 12.")
            d_space = D // 12

            t_monthly = y_true.reshape(12, d_space)
            p_monthly_model = y_pred.reshape(12, d_space)

            kpc_m, info_m = signature_kernel_pc_diag(
                t_monthly, p_monthly_model, m=int(ord_sig),
                base_kernel=base_kernel,
                lmda=float(lmda_kpc),
                include_zero=True
            )

            y_clim = y_clim_concat12_from_clim(
                clim_compressed=clim_compressed,
                wt=wt,
                tau=tau, s=s,
                start_offset=int(start_offset),
                stride=int(stride),
            )
            p_monthly_clim = y_clim.reshape(12, d_space)

            kpc_c, _info_c = signature_kernel_pc_diag(
                t_monthly, p_monthly_clim, m=int(ord_sig),
                base_kernel=base_kernel,
                lmda=float(lmda_kpc),
                include_zero=True
            )

            used_tau += 1

            if np.isfinite(kpc_m):
                kpc_list_m.append(float(kpc_m))
                last_info = info_m
            if np.isfinite(kpc_c):
                kpc_list_c.append(float(kpc_c))

    coverage = (used_tau / tau_total) if tau_total > 0 else 0.0
    if (coverage < float(min_coverage)) or (len(kpc_list_m) == 0):
        mean_m = -np.inf
        std_m  = 0.0
        mean_c = np.nan
        std_c  = 0.0
        if return_diag:
            return (mean_m, std_m, mean_c, std_c, float(coverage),
                    int(len(kpc_list_m)), int(len(kpc_list_c)), last_info, diag)
        return (mean_m, std_m, mean_c, std_c, float(coverage),
                int(len(kpc_list_m)), int(len(kpc_list_c)), last_info)

    mean_m = float(np.mean(kpc_list_m))
    std_m  = float(np.std(kpc_list_m, ddof=1)) if len(kpc_list_m) >= 2 else 0.0

    mean_c = float(np.mean(kpc_list_c)) if len(kpc_list_c) > 0 else np.nan
    std_c  = float(np.std(kpc_list_c, ddof=1)) if len(kpc_list_c) >= 2 else 0.0

    if return_diag:
        return (mean_m, std_m, mean_c, std_c, float(coverage),
                int(len(kpc_list_m)), int(len(kpc_list_c)), last_info, diag)
    return (mean_m, std_m, mean_c, std_c, float(coverage),
            int(len(kpc_list_m)), int(len(kpc_list_c)), last_info)


# =========================
# optimization utilities
# =========================
def scan_LMDA_for_SIG_max_kpc_1d_bounded(
    M0_sig, Xfeat, s, num_rej_fixed,
    lm_min, lm_max,
    lmda0=None,
    ratio_init=1.5,
    maxiter=20,
    eps=1e-12,
    xfeat_mode="concat12",
    base_kernel=None, ord_sig=7,
    lmda_kpc=1.0,
    min_coverage=0.8,
    tau_start=1,
    cv_mode="lfo",
    gap=0,
    verbose_eval=False,
    eval_print_every=5,
):
    """
    returns: (best_mkpc, best_LMDA, coverage, kpc_cnt)
    """
    t0 = time.time()
    nr = int(num_rej_fixed)

    if lmda0 is not None and np.isfinite(lmda0) and lmda0 > 0:
        lo = max(float(lm_min), float(lmda0) / float(ratio_init))
        hi = min(float(lm_max), float(lmda0) * float(ratio_init))
    else:
        lo, hi = float(lm_min), float(lm_max)

    u_lo, u_hi = np.log(lo), np.log(hi)
    eval_cnt = {"n": 0}

    def eval_mkpc(LMDA):
        mkpc, cov, cnt, _ = mean_kpc_score(
            params=[float(LMDA), nr],
            M0=M0_sig, X_all=Xfeat, s=s, eps=eps,
            xfeat_mode=xfeat_mode,
            base_kernel=base_kernel, ord_sig=ord_sig,
            lmda_kpc=float(lmda_kpc),
            min_coverage=min_coverage,
            verbose=False,
            tau_start=tau_start,
            cv_mode=cv_mode,
            gap=gap,
            return_diag=False,
        )
        return float(mkpc), float(cov), int(cnt)

    def obj_u(u):
        eval_cnt["n"] += 1
        LMDA = float(np.exp(u))
        mkpc, cov, cnt = eval_mkpc(LMDA)
        if verbose_eval and (eval_cnt["n"] % int(eval_print_every) == 0):
            print(f"#   [LMDA-1D eval] nr={nr:3d} eval={eval_cnt['n']:3d} LMDA={LMDA:.6g} mkpc={mkpc:.6g}")
            sys.stdout.flush()
        if not np.isfinite(mkpc):
            return 1e100
        return -mkpc

    best = (-np.inf, None, None, None)  # (mkpc, LMDA, cov, cnt)
    try:
        if lmda0 is not None and np.isfinite(lmda0) and lmda0 > 0:
            mk0, cov0, cnt0 = eval_mkpc(float(lmda0))
            if np.isfinite(mk0):
                best = (mk0, float(lmda0), cov0, cnt0)

        res = sp.optimize.minimize_scalar(
            obj_u,
            method="bounded",
            bounds=(u_lo, u_hi),
            options={"maxiter": int(maxiter)},
        )

        LMDA_star = float(np.exp(res.x))
        mkpc_star, cov_star, cnt_star = eval_mkpc(LMDA_star)
        if np.isfinite(mkpc_star) and mkpc_star > best[0]:
            best = (mkpc_star, LMDA_star, cov_star, cnt_star)

        for LM in (lo, hi):
            mk, cov, cnt = eval_mkpc(float(LM))
            if np.isfinite(mk) and mk > best[0]:
                best = (mk, float(LM), cov, cnt)

    except KeyboardInterrupt:
        print(f"# [SIG] nr={nr:3d} interrupted (KeyboardInterrupt). evals={eval_cnt['n']}")
        sys.stdout.flush()
        raise
    except Exception as e:
        print(f"# [SIG] nr={nr:3d} FAILED in 1D optimize. evals={eval_cnt['n']}")
        print("# exception:", repr(e))
        traceback.print_exc()
        sys.stdout.flush()
        return (-np.inf, None, 0.0, 0)

    dt = time.time() - t0
    mkpc_best, LMDA_best, cov_best, cnt_best = best
    print(f"# [SIG] nr={nr:3d} done. evals={eval_cnt['n']} time={dt:.1f}s")
    sys.stdout.flush()

    if LMDA_best is None:
        return (-np.inf, None, 0.0, 0)
    return (float(mkpc_best), float(LMDA_best), float(cov_best), int(cnt_best))


# =========================
# main
# =========================
def main():
    JST = timezone(timedelta(hours=9))
    print("# TIME:", datetime.now(JST).isoformat(timespec="seconds"))
    print("# COMMAND:", " ".join(sys.argv))

    ap = argparse.ArgumentParser()
    ap.add_argument("--cv_mode", choices=["lfo", "lso"], default="lfo",
                    help="lfo: past-only up to (tau-1)->tau. lso: leave-s-out exclusion window.")
    ap.add_argument("--lso_gap", type=int, default=0,
                    help="extra gap for LSO exclusion window (default 0).")

    ap.add_argument("--kernel", choices=["sig", "spk", "both"], default="both")
    ap.add_argument("--s", type=int, default=5)
    ap.add_argument("--ord", type=int, default=7)

    ap.add_argument("--xfeat", choices=["concat12"], default="concat12")
    ap.add_argument("--start_offset", type=int, default=0)

    ap.add_argument("--optimize_sig", action="store_true")
    ap.add_argument("--max_rej", type=int, default=80)
    ap.add_argument("--min_rej", type=int, default=0)
    ap.add_argument("--rej_step", type=int, default=5)

    ap.add_argument("--lmda_min", type=float, default=0.05)
    ap.add_argument("--lmda_max", type=float, default=20.0)

    ap.add_argument("--lmda_refine_ratio", type=float, default=1.25)
    ap.add_argument("--lmda_1d_maxiter", type=int, default=20)

    ap.add_argument("--min_coverage", type=float, default=0.8)

    ap.add_argument("--spk_weight", choices=["cumsum", "nocumsum"], default="cumsum")
    ap.add_argument("--normalize_spk", action="store_true")

    ap.add_argument("--vmax_rms", type=float, default=2.0)
    ap.add_argument("--vmax_drms", type=float, default=0.2)

    ap.add_argument("--save_inputs", action="store_true")

    # kPC settings
    ap.add_argument("--lmda_kpc", type=float, default=1.0)

    args = ap.parse_args()

    tau_start = args.s

    cv_mode = str(args.cv_mode)
    if cv_mode == "lso":
        gap = int(args.lso_gap)
    else:
        gap = 0
        if int(args.lso_gap) != 0:
            print("# NOTE: --lso_gap is ignored because cv_mode=lfo (gap forced to 0).")

    if args.xfeat != "concat12":
        raise ValueError("This script is for --xfeat concat12.")
    if not (0 <= args.start_offset <= 11):
        raise ValueError("--start_offset must be in 0..11")
    if gap < 0:
        raise ValueError("--lso_gap must be >= 0")

    print(f"# CV: mode={cv_mode} gap={gap}")

    # ---- load ----
    sst_compressed = np.load("sst_compressed.npy")      # anomaly, degC
    clim_compressed = np.load("clim_compressed.npy")    # climatology, degC
    valid_indices = np.load("valid_indices.npy")
    wt = compute_flat_area_weights(valid_indices, ny=89, nx=180, normalize=False)
    
    # ---- area-mean RMS correction factor (diagnostic only) ----
    d = int(wt.size)
    sum_w2 = float(np.sum(wt**2))          # = sum_i cos(lat_i) over valid points
    c_area = float(np.sqrt(d / (sum_w2 + 1e-300)))
#    print(f"# area-weight diag: d={d}")
#    print(f"# area-weight diag: sum_w2=sum(wt^2)={sum_w2:.15e}")
#    print(f"# area-RMS correction factor: c_area=sqrt(d/sum_w2)={c_area:.15e}")
    
    # ---- build Xfeat ----
    Xfeat = build_Xfeat(sst_compressed, wt, mode=args.xfeat, stride=12, start_offset=args.start_offset)

    # ---- base kernel for kPC ----
    sigma_data = float(np.load("sigma.npy"))
    print("# sigma (Sig base kernel for kPC) =", sigma_data)
    base_kernel = base_kernel_factory(kernel_type="rbf", sigma=sigma_data)

    # candidate NUM_REJ grid
    num_grid = list(range(int(args.min_rej), int(args.max_rej) + 1, int(args.rej_step)))

    # =========================
    # SPK: optimize NUM_REJ by mean kPC (model only)
    # =========================
    if args.kernel in ("spk", "both"):
        X0w = build_windows_tensor_from_monthly(
            sst_compressed, wt, n=13, stride=12, start_offset=args.start_offset
        )
        sigma_spk = sigma_spk_numpy(X0w, I=12)
        print("# sigma (SPK definition) =", sigma_spk)

        month_w = spk_month_weights(args.spk_weight)
        H_spk = sum_of_pairs_gram_from_windows(
            X0w, sigma=sigma_spk, month_w=month_w, normalize_diag=args.normalize_spk
        )
        M0_spk = [H_spk]

        print(f"# [SPK] maximize mean kPC over NUM_REJ in [{args.min_rej}..{args.max_rej}] step={args.rej_step} (xfeat={args.xfeat})")
        best_spk = (-np.inf, None, None, None)  # (mean_kPC, NUM_REJ, coverage, kpc_cnt)

        for nr in num_grid:
            mkpc, cov, cnt, _, diag = mean_kpc_score(
                params=[0.0, int(nr)],
                M0=M0_spk, X_all=Xfeat, s=args.s,
                xfeat_mode=args.xfeat,
                base_kernel=base_kernel, ord_sig=args.ord,
                lmda_kpc=float(args.lmda_kpc),
                min_coverage=args.min_coverage,
                verbose=False,
                tau_start=tau_start,
                cv_mode=cv_mode,
                gap=gap,
                return_diag=True,
            )
            if not np.isfinite(mkpc):
                print(f"# [SPK] nr={nr:3d} mean_kPC=INVALID (coverage={cov:.3f}, kpc_cnt={cnt})")
                continue
            print(f"# [SPK] nr={nr:3d} mean_kPC={mkpc:.6f} (coverage={cov:.3f}, kpc_cnt={cnt}) | {summarize_mode_diag(diag)}")
            if mkpc > best_spk[0]:
                best_spk = (mkpc, int(nr), cov, cnt)

        if best_spk[1] is None:
            print("# [SPK BEST] NONE (all invalid under min_coverage).")
        else:
            mkpc_spk, nr_spk, cov_spk, cnt_spk = best_spk
            print(f"# [SPK BEST] NUM_REJ={nr_spk} mean_kPC={mkpc_spk:.6f} (coverage={cov_spk:.3f}, kpc_cnt={cnt_spk})")

            (mkpc_m, std_m, mkpc_c, std_c, cov2, cnt_m, cnt_c, last_info, diag_best) = \
                mean_kpc_score_pair_model_clim(
                    params=[0.0, nr_spk],
                    M0=M0_spk, X_all=Xfeat, s=args.s,
                    xfeat_mode=args.xfeat,
                    base_kernel=base_kernel, ord_sig=args.ord,
                    lmda_kpc=float(args.lmda_kpc),
                    min_coverage=args.min_coverage,
                    tau_start=tau_start,
                    cv_mode=cv_mode,
                    gap=gap,
                    clim_compressed=clim_compressed, wt=wt,
                    start_offset=args.start_offset, stride=12,
                    return_diag=True,
                )
            print(f"# [SPK BEST DIAG] {summarize_mode_diag(diag_best)}")
            if last_info is not None:
                print("# [SPK kPC] per-order weighted TP (last tau):", last_info["TP_wlevels"])
            print(f"# [SPK kPC] mean_kPC_model={mkpc_m:.6f} std_kPC_model={std_m:.6f} (cnt={cnt_m})")
            print(f"# [SPK kPC] mean_kPC_clim ={mkpc_c:.6f} std_kPC_clim ={std_c:.6f} (cnt={cnt_c})")

            (mse_m, rms_m, std_rms_m,
             mse_p, rms_p, std_rms_p,
             mse_c, rms_c, std_rms_c) = mse_rms_error(
                params=[0.0, nr_spk], M0=M0_spk, X_all=Xfeat, s=args.s,
                xfeat_mode=args.xfeat, min_coverage=args.min_coverage, verbose=True,
                tau_start=tau_start, cv_mode=cv_mode, gap=gap,
                clim_compressed=clim_compressed,
                wt=wt,
                start_offset=args.start_offset,
                stride=12,
                return_std=True,
            )
            # ---- overwrite to "area-normalized RMS" as the official RMS ----
            mse_m *= c_area**2
            mse_p *= c_area**2
            mse_c *= c_area**2
            rms_m *= c_area
            rms_p *= c_area
            rms_c *= c_area
            std_rms_m *= c_area
            std_rms_p *= c_area
            std_rms_c *= c_area
            
            print(f"# [SPK RMS] model: MSE={mse_m:.6e}, RMS={rms_m:.6e}, RMS_std={std_rms_m:.6e}")
            print(f"# [SPK RMS] pers : MSE={mse_p:.6e}, RMS={rms_p:.6e}, RMS_std={std_rms_p:.6e}")
            print(f"# [SPK RMS] clim : MSE={mse_c:.6e}, RMS={rms_c:.6e}, RMS_std={std_rms_c:.6e}")

            (_, rms_x_maps) = loo_spatial_rms_maps(
                params=[0.0, nr_spk], M0=M0_spk, X_all=Xfeat,
                valid_indices=valid_indices, wt=wt,
                ny=89, nx=180, s=args.s, xfeat_mode=args.xfeat,
                tau_start=tau_start, cv_mode=cv_mode, gap=gap,
                clim_compressed=clim_compressed,
                start_offset=args.start_offset,
                stride=12,
            )
            rms_x_model_map, rms_x_pers_map, rms_x_clim_map = rms_x_maps

            vmin_rms, vmax_rms = 0.0, float(args.vmax_rms)
            tag = f"s{args.s}_{args.xfeat}_off{args.start_offset}_{cv_mode}g{gap}"
            plot_map_pdf(rms_x_model_map, f"loo_rms_map_spk_{tag}_model_degC.pdf",
                         title=f"LOO RMS (SPK model, s={args.s}) [{args.xfeat}] {cv_mode} g={gap} [degC]",
                         vmin=vmin_rms, vmax=vmax_rms, cmap="viridis")
            plot_map_pdf(rms_x_pers_map, f"loo_rms_map_spk_{tag}_pers_degC.pdf",
                         title=f"LOO RMS (persistence, s={args.s}) [{args.xfeat}] {cv_mode} g={gap} [degC]",
                         vmin=vmin_rms, vmax=vmax_rms, cmap="viridis")
            plot_map_pdf(rms_x_clim_map, f"loo_rms_map_spk_{tag}_clim_degC.pdf",
                         title=f"LOO RMS (climatology, s={args.s}) [{args.xfeat}] {cv_mode} g={gap} [degC]",
                         vmin=vmin_rms, vmax=vmax_rms, cmap="viridis")

            drms_x = rms_x_clim_map - rms_x_model_map
            vmax_d = float(args.vmax_drms)
            plot_map_pdf(drms_x, f"loo_drms_spk_{tag}_clim-model_degC.pdf",
                         title=f"LOO dRMS (clim-model, s={args.s}) [{args.xfeat}] {cv_mode} g={gap} [degC]",
                         vmin=-vmax_d, vmax=vmax_d, cmap="RdBu_r")

            if args.save_inputs:
                np.savez("koopman_inputs_spk.npz",
                         M0=np.array(M0_spk, dtype=object),
                         X=Xfeat,
                         params=np.array([0.0, float(nr_spk)], dtype=float),
                         meta=np.array([args.xfeat, str(args.start_offset), args.spk_weight, cv_mode, int(gap)], dtype=object))
                print("# [SAVED] koopman_inputs_spk.npz")

    # =========================
    # SIG: optimize (NUM_REJ, LMDA) by mean kPC (model only)
    # =========================
    if args.kernel in ("sig", "both"):
        data = np.load("Mw_orders.npz", allow_pickle=True)
        ORD = int(args.ord)
        M0_sig = [data[f"M_order{i}"] for i in range(ORD + 1)]

        if not args.optimize_sig:
            LMDA_best = 1.0
            num_rej_best = 0
            print(f"# [SIG] (no optimize) LMDA={LMDA_best:.6f}, NUM_REJ={num_rej_best}")
        else:
            print(f"# [SIG] maximize mean kPC over NUM_REJ in [{args.min_rej}..{args.max_rej}] step={args.rej_step} and LMDA (1D bounded on log-LMDA)")
            sys.stdout.flush()

            best_all = (-np.inf, None, None, None, None)  # (mkpc, nr, LMDA, cov, cnt)

            for nr in num_grid:
                best_1d = scan_LMDA_for_SIG_max_kpc_1d_bounded(
                    M0_sig=M0_sig, Xfeat=Xfeat, s=args.s, num_rej_fixed=nr,
                    lm_min=args.lmda_min, lm_max=args.lmda_max,
                    lmda0=float(args.lmda_kpc),                    # 初期値 = LMDA_kPC
                    ratio_init=float(args.lmda_refine_ratio),
                    maxiter=int(args.lmda_1d_maxiter),
                    xfeat_mode=args.xfeat,
                    base_kernel=base_kernel, ord_sig=ORD,
                    lmda_kpc=float(args.lmda_kpc),
                    min_coverage=args.min_coverage,
                    tau_start=tau_start,
                    cv_mode=cv_mode,
                    gap=gap,
                    verbose_eval=False,
                    eval_print_every=5,
                )
                mkpc, LMDA_star, cov, cnt = best_1d
                if LMDA_star is None:
                    print(f"# [SIG] nr={nr:3d} -> INVALID (1D failed or no finite mkpc).")
                    sys.stdout.flush()
                    continue

                print(f"# [SIG] nr={nr:3d} best LMDA={LMDA_star:.10f} mean_kPC={mkpc:.6f} (coverage={cov:.3f}, kpc_cnt={cnt})")
                sys.stdout.flush()

                if mkpc > best_all[0]:
                    best_all = (mkpc, int(nr), float(LMDA_star), float(cov), int(cnt))

            if best_all[1] is None:
                print("# [SIG BEST] NONE (all invalid under min_coverage).")
                LMDA_best = 1.0
                num_rej_best = 0
            else:
                mkpc_sig, num_rej_best, LMDA_best, cov_best, cnt_best = best_all
                print(f"# [SIG BEST] NUM_REJ={num_rej_best} LMDA={LMDA_best:.10f} mean_kPC={mkpc_sig:.6f} "
                      f"(coverage={cov_best:.3f}, kpc_cnt={cnt_best})")

        (mkpc_m, std_m, mkpc_c, std_c, _cov, cnt_m, cnt_c, last_info, diag_sig) = \
            mean_kpc_score_pair_model_clim(
                params=[LMDA_best, num_rej_best],
                M0=M0_sig, X_all=Xfeat, s=args.s,
                xfeat_mode=args.xfeat,
                base_kernel=base_kernel, ord_sig=ORD,
                lmda_kpc=float(args.lmda_kpc),
                min_coverage=args.min_coverage,
                tau_start=tau_start,
                cv_mode=cv_mode,
                gap=gap,
                clim_compressed=clim_compressed, wt=wt,
                start_offset=args.start_offset, stride=12,
                return_diag=True,
            )
        print(f"# [SIG BEST DIAG] {summarize_mode_diag(diag_sig)}")
        if last_info is not None:
            print("# [SIG kPC] per-order weighted TP (last tau):", last_info["TP_wlevels"])
        print(f"# [SIG kPC] mean_kPC_model={mkpc_m:.6f} std_kPC_model={std_m:.6f} (cnt={cnt_m})")
        print(f"# [SIG kPC] mean_kPC_clim ={mkpc_c:.6f} std_kPC_clim ={std_c:.6f} (cnt={cnt_c})")

        (mse_m, rms_m, std_rms_m,
         mse_p, rms_p, std_rms_p,
         mse_c, rms_c, std_rms_c) = mse_rms_error(
            params=[LMDA_best, num_rej_best], M0=M0_sig, X_all=Xfeat, s=args.s,
            xfeat_mode=args.xfeat, min_coverage=args.min_coverage, verbose=True,
            tau_start=tau_start, cv_mode=cv_mode, gap=gap,
            clim_compressed=clim_compressed,
            wt=wt,
            start_offset=args.start_offset,
            stride=12,
            return_std=True,
        )
        # ---- overwrite to "area-normalized RMS" as the official RMS ----
        mse_m *= c_area**2
        mse_p *= c_area**2
        mse_c *= c_area**2
        rms_m *= c_area
        rms_p *= c_area
        rms_c *= c_area
        std_rms_m *= c_area
        std_rms_p *= c_area
        std_rms_c *= c_area
        print(f"# [SPK RMS] model: MSE={mse_m:.6e}, RMS={rms_m:.6e}, RMS_std={std_rms_m:.6e}")
        print(f"# [SPK RMS] pers : MSE={mse_p:.6e}, RMS={rms_p:.6e}, RMS_std={std_rms_p:.6e}")
        print(f"# [SPK RMS] clim : MSE={mse_c:.6e}, RMS={rms_c:.6e}, RMS_std={std_rms_c:.6e}")

        (_, rms_x_maps) = loo_spatial_rms_maps(
            params=[LMDA_best, num_rej_best],
            M0=M0_sig, X_all=Xfeat,
            valid_indices=valid_indices, wt=wt,
            ny=89, nx=180, s=args.s, xfeat_mode=args.xfeat,
            tau_start=tau_start, cv_mode=cv_mode, gap=gap,
            clim_compressed=clim_compressed,
            start_offset=args.start_offset,
            stride=12,
        )
        rms_x_model_map, rms_x_pers_map, rms_x_clim_map = rms_x_maps

        vmin_rms, vmax_rms = 0.0, float(args.vmax_rms)
        tag = f"s{args.s}_{args.xfeat}_off{args.start_offset}_{cv_mode}g{gap}"
        plot_map_pdf(rms_x_model_map, f"loo_rms_map_sig_{tag}_model_degC.pdf",
                     title=f"LOO RMS (SIG model, s={args.s}) [{args.xfeat}] {cv_mode} g={gap} [degC]",
                     vmin=vmin_rms, vmax=vmax_rms, cmap="viridis")
        plot_map_pdf(rms_x_pers_map, f"loo_rms_map_sig_{tag}_pers_degC.pdf",
                     title=f"LOO RMS (persistence, s={args.s}) [{args.xfeat}] {cv_mode} g={gap} [degC]",
                     vmin=vmin_rms, vmax=vmax_rms, cmap="viridis")
        plot_map_pdf(rms_x_clim_map, f"loo_rms_map_sig_{tag}_clim_degC.pdf",
                     title=f"LOO RMS (climatology, s={args.s}) [{args.xfeat}] {cv_mode} g={gap} [degC]",
                     vmin=vmin_rms, vmax=vmax_rms, cmap="viridis")

        drms_x = rms_x_clim_map - rms_x_model_map
        vmax_d = float(args.vmax_drms)
        plot_map_pdf(drms_x, f"loo_drms_sig_{tag}_clim-model_degC.pdf",
                     title=f"LOO dRMS (clim-model, s={args.s}) [{args.xfeat}] {cv_mode} g={gap} [degC]",
                     vmin=-vmax_d, vmax=vmax_d, cmap="RdBu_r")

        if args.save_inputs:
            np.savez("koopman_inputs_sig.npz",
                     M0=np.array(M0_sig, dtype=object),
                     X=Xfeat,
                     params=np.array([float(LMDA_best), float(num_rej_best)], dtype=float),
                     meta=np.array([args.xfeat, str(args.start_offset), cv_mode, int(gap)], dtype=object))
            print("# [SAVED] koopman_inputs_sig.npz")


if __name__ == "__main__":
    main()
    
