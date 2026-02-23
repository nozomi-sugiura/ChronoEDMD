#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read Koopman inputs (*.npz) produced by LOO_test3.py and plot Koopman modes.

Fixes included:
  1) Xfeat=concat12 (shape: 12*d) から得られる Xi (shape: r x 12*d) を
     「年平均」(shape: r x d) に変換してから空間マップへ配置する。
     → これにより shape mismatch を解消します。
  2) 面積重み w = sqrt(cos(lat)) は “学習座標 z = w*a” のために用いている前提を維持し、
     物理空間(degC)への戻しは「描画時のみ」 a = z / w で行う。

Usage:
  python -u read_koopman_fixed.py <smon> [koopman_inputs_sig.npz]
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cmocean

# =========================================================
# Utilities
# =========================================================
def _sigma_max_power(A, v0=None, iters=40, tol=1e-7):
    """
    Approximate spectral norm ||A||_2 (largest singular value) by power iteration
    on A^*A. Works for real/complex dense matrices.

    Returns
    -------
    sigma : float
        Approximation of ||A||_2
    v : ndarray (n,)
        Approximate dominant right singular vector (warm-start usable)
    """
    n = A.shape[1]
    if v0 is None:
        v = np.random.default_rng(0).standard_normal(n) + 1j*np.random.default_rng(1).standard_normal(n)
    else:
        v = np.array(v0, dtype=A.dtype, copy=True)

    # normalize
    nv = np.linalg.norm(v)
    if nv == 0:
        v = np.ones(n, dtype=A.dtype)
        nv = np.linalg.norm(v)
    v /= nv

    sigma_old = 0.0
    for _ in range(int(iters)):
        Av = A @ v
        w = A.conj().T @ Av  # (A^*A)v
        nw = np.linalg.norm(w)
        if nw == 0:
            return 0.0, v
        v = w / nw

        sigma = np.linalg.norm(A @ v)
        if abs(sigma - sigma_old) <= tol * max(1.0, sigma):
            break
        sigma_old = sigma

    return float(sigma), v


def transient_growth_norm2(K, nmax=200, iters=40, tol=1e-7, return_powers=False):
    """
    Compute/approximate g(n)=||K^n||_2 for n=0..nmax with low cost.

    Parameters
    ----------
    K : (n,n) array_like
        Koopman matrix (real/complex).
    nmax : int
        Max power n.
    iters, tol : power-iteration controls for ||A||_2 estimation.
    return_powers : bool
        If True, also return the list of K^n matrices (memory heavy).

    Returns
    -------
    ns : ndarray (nmax+1,)
    g  : ndarray (nmax+1,)
        g[n] ≈ ||K^n||_2
    gmax : float
        max_n g[n]
    n_star : int
        argmax_n g[n]
    (optional) powers : list of ndarray
        powers[n] = K^n
    """
    K = np.asarray(K)
    n = K.shape[0]
    if K.shape[0] != K.shape[1]:
        raise ValueError("K must be square.")

    ns = np.arange(nmax + 1, dtype=int)
    g = np.empty(nmax + 1, dtype=float)
    g[0] = 1.0

    A = np.eye(n, dtype=K.dtype)  # A = K^0
    v = None  # warm start (dominant right singular vector)

    powers = [A.copy()] if return_powers else None

    for k in range(1, nmax + 1):
        A = K @ A  # A = K^k
        sigma, v = _sigma_max_power(A, v0=v, iters=iters, tol=tol)
        g[k] = sigma
        if return_powers:
            powers.append(A.copy())

    n_star = int(np.nanargmax(g))
    gmax = float(g[n_star])

    if return_powers:
        return ns, g, gmax, n_star, powers
    return ns, g, gmax, n_star


def kreiss_constant_discrete(
    K,
    norm="2",
    eps=1e-6,
    r_max=2.0,
    n_r=50,
    n_theta=720,
    refine_levels=1,
    refine_factor=4,
    return_argmax=False,
):
    r"""
    Approximate discrete-time Kreiss constant:
        kappa(K) = sup_{|z|>1} (|z|-1) ||(zI - K)^{-1}||.

    If spectral radius > 1, returns inf.
    norm: currently supports "2","fro","1","inf".
    """
    import scipy.linalg as la

    K = np.asarray(K)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix.")

    eigvals = np.linalg.eigvals(K)
    if np.max(np.abs(eigvals)) > 1 + 1e-12:
        return (np.inf, None) if return_argmax else np.inf

    n = K.shape[0]
    I = np.eye(n, dtype=complex)

    def opnorm_inv(z):
        M = (z * I - K)
        X = la.solve(M, I, assume_a="gen", check_finite=False)
        if norm == "2":
            s = la.svdvals(X, check_finite=False)
            return float(np.max(s))
        elif norm == "fro":
            return float(la.norm(X, ord="fro"))
        elif norm == "1":
            return float(la.norm(X, ord=1))
        elif norm == "inf":
            return float(la.norm(X, ord=np.inf))
        else:
            raise ValueError(f"Unknown norm: {norm}")

    radii = np.geomspace(1.0 + eps, r_max, n_r)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    best_val = -np.inf
    best_z = None

    def scan(radii_local, thetas_local):
        nonlocal best_val, best_z
        for r in radii_local:
            zs = r * np.exp(1j * thetas_local)
            for z in zs:
                try:
                    invn = opnorm_inv(z)
                except Exception:
                    continue
                val = (abs(z) - 1.0) * invn
                if val > best_val:
                    best_val = float(val)
                    best_z = z

    # coarse
    scan(radii, thetas)

    # refine around best (optional)
    for lvl in range(refine_levels):
        if best_z is None or not np.isfinite(best_val):
            break
        r0 = abs(best_z)
        th0 = np.angle(best_z)

        r_lo = max(1.0 + eps, r0 / 1.5)
        r_hi = min(r_max,      r0 * 1.5)

        dth = (2.0 * np.pi) / n_theta
        th_lo = th0 - 8 * dth
        th_hi = th0 + 8 * dth

        n_r_loc  = max(30, int(refine_factor * n_r / (2 ** (lvl + 1))))
        n_th_loc = max(180, int(refine_factor * n_theta / (2 ** (lvl + 1))))

        radii_loc  = np.geomspace(r_lo, r_hi, n_r_loc)
        thetas_loc = np.linspace(th_lo, th_hi, n_th_loc, endpoint=False)
        scan(radii_loc, thetas_loc)

    if return_argmax:
        return best_val, best_z
    return best_val

def save_gram_matrix(G, filename="G_sst.pdf",
                     title="Gram Matrix Heatmap",
                     label=r"$\hat{G}_{ij}$",
                     vmin=None, vmax=None):
    n = G.shape[0]
    years = np.arange(n) + 1854
    fig, ax = plt.subplots(figsize=(7, 7))

    cmap = plt.get_cmap("coolwarm", 512)
    cax = ax.matshow(G, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    tick_positions = np.linspace(0, n - 1, min(10, n), dtype=int)
    tick_labels = years[tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel("year $j$")
    ax.set_ylabel("year $i$")

    fig.colorbar(cax, label=label)
    ax.set_title(title)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def compute_flat_area_weights(valid_indices, ny=89, nx=180):
    """
    w = sqrt(cos(lat)) を valid_indices のみに返す
    """
    latitudes = np.linspace(-88, 88, ny)
    coslat = np.cos(np.deg2rad(latitudes))  # (ny,)
    weights_2d = np.repeat(coslat[:, np.newaxis], nx, axis=1)  # (ny,nx)
    weights_flat = weights_2d.reshape(-1)  # (ny*nx,)
    return np.sqrt(weights_flat[valid_indices])


def normalize_eigenfunctions(Phi_x):
    a = np.sqrt(np.linalg.norm(Phi_x, axis=0) ** 2 / Phi_x.shape[0])
    for i in range(Phi_x.shape[1]):
        Phi_x[:, i] /= a[i]
    return Phi_x


def efld_prd(eigval):
    log_eig = np.log(eigval)
    efold = 1 / log_eig.real if log_eig.real != 0.0 else np.inf
    period = np.abs(2 * np.pi / log_eig.imag) if log_eig.imag != 0.0 else np.inf
    return efold, period


def residual2(K2, L, Tl):
    res = np.zeros(L.shape[0])
    for j in range(L.shape[0]):
        res0 = Tl[:, j].T.conj() @ Tl[:, j]
        res1 = Tl[:, j].T.conj() @ K2 @ Tl[:, j]
        val = (res1.real / res0.real) - np.abs(L[j]) ** 2
        res[j] = np.sqrt(max(0.0, float(val)))
    return res

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


def plot_mode_timeseries(Phi_x1, L1, jt, E, dt, smon,
                         plot_kind="raw",   # "raw" or "proc"
                         theta=None,
                         csv_save=True, csv_path=None,
                         year_shift_plot=0.5):  # 表示用のシフト（年平均窓の中心なら +0.5）
    """
    Phi_x1 : complex eigenfunction values a_j(t) (length n)
    L1     : eigenvalue (discrete-time, step dt)
    dt     : sampling interval [yr]
    smon   : start month of first record (1..12)

    - raw  : a_raw = Phi_x1
    - proc : a_proc = deamp(a_raw) and then optional rotation by theta
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    n = Phi_x1.shape[0]
    t = np.arange(n, dtype=float) * dt

    # ---- time axis (definition): month-start convention ----
    years = 1854.0 + (smon - 1.0) / 12.0 + t  # 月初（定義用）
    years_plot = years + float(year_shift_plot)  # 表示用（年窓の中央へ寄せる等）

    # ---- eigenvalue to continuous-time rate ----
    lam = (np.log(L1) / dt)   # sigma + i omega
    sigma = float(lam.real)
    omega = float(lam.imag)

    # ---- raw coefficient ----
    a_raw = Phi_x1.astype(complex)

    # ---- processed coefficient (deamp + optional rotation) ----
    a_proc = a_raw.copy()
    if np.isfinite(sigma) and abs(sigma) > 0:
        t_ref = t.mean()
        a_proc = a_proc * np.exp(-sigma * (t - t_ref))  # deamp
    if theta is not None:
        a_proc = a_proc * np.exp(-1j * float(theta))    # rotate

    # ---- choose what to plot ----
    if plot_kind == "raw":
        a_plot = a_raw
        tag = " (raw)"
    elif plot_kind == "proc":
        a_plot = a_proc
        tag = " (proc)"
        if theta is not None:
            tag += " (rot)"
        tag += " (deamp)"
    else:
        raise ValueError("plot_kind must be 'raw' or 'proc'.")

    real_part = a_plot.real
    imag_part = a_plot.imag

    # ---- period text ----
    if omega != 0.0:
        period = abs(2 * np.pi / omega)
        efold = (1.0 / sigma) if sigma != 0.0 else np.inf
        period_text = f" e-folding: {efold:.2f} yr, Period: {period:.2f} yr"
    else:
        efold = (1.0 / sigma) if sigma != 0.0 else np.inf
        period_text = f" e-folding: {efold:.2f} yr, Period: ∞ yr"

    # ---- CSV: save BOTH raw and proc (always unambiguous) ----
    if csv_save:
        os.makedirs("frames", exist_ok=True)
        if csv_path is None:
            csv_path = f"frames/timeseries_{jt:03d}.csv"

        def pack(a):
            amp = np.abs(a)
            ph = np.angle(a)
            return amp, ph, np.unwrap(ph)

        amp_raw, ph_raw, phu_raw = pack(a_raw)
        amp_prc, ph_prc, phu_prc = pack(a_proc)

        df = pd.DataFrame({
            "year": years,               # 定義用（窓の開始）
            "year_plot": years_plot,     # 表示用（窓の中心など）

            "real_raw": a_raw.real,
            "imag_raw": a_raw.imag,
            "amplitude_raw": amp_raw,
            "phase_rad_raw": ph_raw,
            "phase_unwrapped_rad_raw": phu_raw,

            "real_proc": a_proc.real,
            "imag_proc": a_proc.imag,
            "amplitude_proc": amp_prc,
            "phase_rad_proc": ph_prc,
            "phase_unwrapped_rad_proc": phu_prc,

            "sigma": np.full(n, sigma),
            "omega": np.full(n, omega),
            "theta_rad": np.full(n, float(theta) if theta is not None else np.nan),
        })
        df.to_csv(csv_path, index=False, float_format="%.6f")

    # ---- plot ----
    plt.figure(figsize=(10, 4))
    plt.plot(years_plot, real_part, label="Real", color="red")
    plt.plot(years_plot, imag_part, label="Imag", color="purple", linestyle="--")
    plt.xlabel("Year")
    plt.ylabel("Eigenfunction value")
    plt.title(f"Time Series of Mode {jt}{tag}{period_text} Energy {E:.2e}", fontsize=8)
    plt.legend()
    plt.grid(True)
    plt.xlim(years_plot[0], years_plot[-1])
    plt.ylim(-1.5, 1.5)
    os.makedirs("frames", exist_ok=True)
    plt.savefig(f"frames/timeseries_{jt:03d}.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    

# =========================================================
# Core
# =========================================================
def koopman_modes(params, M0, X0, valid_indices,
                 dt=1.0, smon=1,
                 nmon=12,
                 plot=True, fname=None):
    """
    X0: shape (1, M, 12*d)  (concat12)
    return: eigvals_keep, Xi_raw, Phi_x, keep_indices, K, Kc
      - Xi_raw is the raw Xi in concat12 coordinates (r x 12*d)
    """
    LMDA, num_rej = params
    NUM_REJ = int(round(num_rej))

    # build H
    H = M0[0].copy()
    for i in range(1, len(M0)):
        H += M0[i] * LMDA ** (2 * i)

    # standard EDMD matrices (yearly transitions)
    G = H[:-1, :-1]
    A = H[1:, :-1]
    L = H[1:, 1:]

    S2, Q = np.linalg.eigh(G)
    S2 = np.maximum(S2, 0.0)
    S = np.diag(np.sqrt(S2))
    Sinv = np.linalg.pinv(S)

    K = Sinv @ Q.T @ A @ Q @ Sinv
    Kc = Q @ Sinv @ Sinv @ Q.T @ A

    eigvals, Vl, V = sp.linalg.eig(K, left=True)
    K2 = Sinv @ Q.T @ L @ Q @ Sinv
    res = residual2(K2, eigvals, Vl)

    sorted_groups = sorted_pair_indices(res, eigvals)

    if NUM_REJ == 0:
        groups_to_keep = sorted_groups
    elif NUM_REJ >= len(sorted_groups):
        groups_to_keep = []
    else:
        groups_to_keep = sorted_groups[:len(sorted_groups) - NUM_REJ]

    keep_indices = sorted([i for group in groups_to_keep for i in group])

    eigvals_keep = eigvals[keep_indices]
    eigvals_drop = eigvals[~np.isin(np.arange(len(eigvals)), keep_indices)]

    # eigenfunction values
    Phi_x = Q @ S @ V[:, keep_indices]
    Phi_x = normalize_eigenfunctions(Phi_x)

    # --- Xi in concat12 space ---
    # X0[0] shape (M, 12*d). Use transitions: states 0..M-2
    X_train_states = X0[0][:-1, :]  # (M-1, 12*d)
    Xi_raw = np.linalg.pinv(Phi_x) @ X_train_states  # (r, 12*d)

    # --- convert to annual mean in weighted space: (r,12,d)->(r,d) ---
    r, D = Xi_raw.shape
    d = len(valid_indices)
    if D != nmon * d:
        raise ValueError(
            f"Xi_raw has D={D}, but expected D=nmon*d={nmon*d} "
            f"(nmon={nmon}, d={d})."
        )
    Xi_w_ann = Xi_raw.reshape(r, nmon, d).mean(axis=1)  # (r,d), still weighted (z-space)

    if plot:
        rank = np.linalg.matrix_rank(Phi_x)
        n_modes = Phi_x.shape[1]
        print(f"#Rank = {rank}, Number of modes = {n_modes}")
        print("#Linear Independence of eigenfunction values?", rank == n_modes)

        ny, nx = 89, 180

        # Map modes to full grid, still weighted (z-space)
        Fld_w = np.ma.masked_all((Xi_w_ann.shape[0], ny * nx), dtype=np.complex64)
        Fld_w[:, valid_indices] = Xi_w_ann

        # map original eigen-index -> index in kept list
        eig_to_fld = {orig_idx: new_idx for new_idx, orig_idx in enumerate(keep_indices)}

        # weights: w=sqrt(cos) on valid grid (for both phys conversion and cos-area energy)
        wt = compute_flat_area_weights(valid_indices, ny=ny, nx=nx)  # (d,)
        cos_w = wt ** 2

        # loop over groups (pairs or singles)
        for group in groups_to_keep:
            if len(group) == 2:
                i, j = group
                efold, period = efld_prd(eigvals[i])

                # weighted-space combined mode (annual-mean)
                Xi_sum_w = (Fld_w[eig_to_fld[i]] + Fld_w[eig_to_fld[j]].conj()) / np.sqrt(2)

                # phys-space (degC) conversion ONLY for plotting/energy:
                Xi_sum_phys = np.ma.masked_all_like(Xi_sum_w)
                Xi_sum_phys[valid_indices] = Xi_sum_w[valid_indices] / wt

                Xi_valid_phys = Xi_sum_phys[valid_indices]  # (d,)
                Energy = np.sum(cos_w * np.abs(Xi_valid_phys) ** 2) / np.sum(cos_w)

                title_real = (f"Re[#{i}+#{j}]: Energy {Energy:.2e}$\\mathrm{{K}}^2$"
                              f"\n efld {efold:.2f} per. {period:.2f}")
                title_imag = (f"Im[#{i}-#{j}]: Energy {Energy:.2e}$\\mathrm{{K}}^2$"
                              f"\n efld {efold:.2f} per. {period:.2f}")

                jt = eig_to_fld[i]
#                print("#Per.", i, period, efold, Energy, np.abs(eigvals[i]))
                plot_mode_timeseries(Phi_x[:, jt], eigvals_keep[jt], i, Energy, dt, smon)
                res_g = float(res[i] + res[j])   # sorted_pair_indices と同じ定義（和）
                print(f"#Per. {i:03d} per= {period:.2f} efld= {efold:.2f} "
                      f"res= {res_g:.3f} E= {Energy:.3e} |lam|= {abs(eigvals[i]):.6f}")
            else:
                i = group[0]
                efold, period = efld_prd(eigvals[i])

                Xi_sum_w = Fld_w[eig_to_fld[i]]

                Xi_sum_phys = np.ma.masked_all_like(Xi_sum_w)
                Xi_sum_phys[valid_indices] = Xi_sum_w[valid_indices] / wt

                Xi_valid_phys = Xi_sum_phys[valid_indices]
                Energy = np.sum(cos_w * np.abs(Xi_valid_phys) ** 2) / np.sum(cos_w)

                title_real = (f"Re[#{i}] {Energy:.2e}$\\mathrm{{K}}^2$"
                              f"\n efld {efold:.2f} per. {period:.2f}")
                title_imag = (f"Im[#{i}] {Energy:.2e}$\\mathrm{{K}}^2$"
                              f"\n efld {efold:.2f} per. {period:.2f}")

                jt = eig_to_fld[i]
                res_g = float(res[i])
#                print("#Per.", i, period, efold, Energy, np.abs(eigvals[i]))
                print(f"#Per. {i:03d} per= {period:.2f} efld= {efold:.2f} "
                      f"res= {res_g:.3f} E= {Energy:.3e} |lam|= {abs(eigvals[i]):.6f}")
                plot_mode_timeseries(Phi_x[:, jt], eigvals_keep[jt], i, Energy, dt, smon)

            # --- draw maps (phys-space annual mean) ---
            real_part = Xi_sum_phys.real.reshape(ny, nx)
            imag_part = Xi_sum_phys.imag.reshape(ny, nx)

            vmax = 0.9 * max(np.abs(real_part).max(), np.abs(imag_part).max())
            vmin = -vmax

            xticks = np.linspace(0, nx - 1, 7, dtype=int)
            yticks = np.linspace(0, ny - 1, 5, dtype=int)
            xticklabels = np.linspace(0, 360, 7).astype(int)
            yticklabels = np.linspace(-90, 90, 5).astype(int)

            fig = plt.figure(figsize=(12, 4))
            spec = gridspec.GridSpec(ncols=3, nrows=1, width_ratios=[1, 1, 0.05])

            ax0 = fig.add_subplot(spec[0])
            ax1 = fig.add_subplot(spec[1])
            cax = fig.add_subplot(spec[2])

            cmap = cmocean.cm.balance.copy()
            cmap.set_bad(color="#654321")

            im0 = ax0.imshow(real_part, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax0.set_title(title_real)
            ax0.set_xticks(xticks, labels=xticklabels, fontsize=9)
            ax0.set_yticks(yticks, labels=yticklabels, fontsize=9)

            im1 = ax1.imshow(imag_part, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
            ax1.set_title(title_imag)
            ax1.set_xticks(xticks, labels=xticklabels, fontsize=9)
            ax1.set_yticks(yticks, labels=yticklabels, fontsize=9)

            cbar = fig.colorbar(im1, cax=cax)
            cbar.set_label("Amplitude (degC, annual mean)")

            fig.subplots_adjust(wspace=0.12, right=0.88)
            os.makedirs("frames", exist_ok=True)
            filename_prefix = "frames/koopman_modes"
            plt.savefig(f"{filename_prefix}_{i:03d}.pdf", dpi=300, bbox_inches="tight")

            # save per-mode data (phys, annual mean)
            np.savez(f"{filename_prefix}_{i:03d}",
                     Xi_data=Xi_sum_phys.data,
                     Xi_mask=Xi_sum_phys.mask,
                     eigv=eigvals[i])

            plt.close(fig)

        # eigenvalue plot
        plt.figure(figsize=(5, 5))
        plt.scatter(eigvals_keep.real, eigvals_keep.imag, color="blue", alpha=0.7,
                    label=f"Accepted ({len(eigvals_keep)})")
        plt.scatter(eigvals_drop.real, eigvals_drop.imag, color="red", alpha=0.5,
                    label=f"Rejected ({len(eigvals_drop)})")
        circle = plt.Circle((0, 0), 1.0, color="gray", fill=False, linestyle="--", label="Unit circle")
        plt.gca().add_artist(circle)
        plt.xlabel("Real part")
        plt.ylabel("Imaginary part")
        plt.grid(True)
        plt.gca().set_aspect("equal")
        plt.title(f"Koopman Eigenvalues\nLMDA={LMDA:.3f}, NUM_REJ={NUM_REJ}")
        plt.legend()
        if fname:
            plt.savefig(fname, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    return eigvals_keep, Xi_raw, Phi_x, keep_indices, K, Kc


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    dt = 1.0
    smon = int(sys.argv[1])  # start month (1..12)
    npz_path = sys.argv[2] if len(sys.argv) >= 3 else "koopman_inputs_sig.npz"

    data = np.load(npz_path, allow_pickle=True)
    valid_indices = np.load("valid_indices.npy")

    M0_raw = data["M0"]  # dtype=object array
    X = np.asarray(data["X"], dtype=np.float64)
    params = np.asarray(data["params"], dtype=np.float64)

    # restore M0 as list of float64 2D arrays
    M0 = [np.asarray(M0_raw[i], dtype=np.float64) for i in range(len(M0_raw))]
    print(len(M0), M0[0].shape)
    
    print("# params", float(params[0]), int(round(params[1])))
    print("# FILE:", npz_path)

    eigvals_keep, Xi_raw, Phi_x, keep_indices, K, Kc = koopman_modes(
        params, M0, X, valid_indices,
        dt=dt, smon=smon,
        plot=True, fname="koopman_eigvals.pdf"
    )

    # -----------------------------
    # Kreiss constant of Koopman matrix K
    # -----------------------------
    kappa, zstar = kreiss_constant_discrete(
    K, norm="2",
    eps=1e-6, r_max=1.15, n_r=20, n_theta=180,
    refine_levels=0, return_argmax=True
    )
    print(f"# Kreiss kappa(K) ≈ {kappa:.6e}  (argmax z ≈ {zstar})")
    # ---- 使用例（あなたのコードの末尾、K ができた後に追記）----
    nmax = 10
    ns, g, gmax, n_star = transient_growth_norm2(K, nmax=nmax, iters=50, tol=1e-8)
    print(f"# transient growth: max ||K^n||_2 over n=0..{nmax} is {gmax:.6e} at n={n_star}")
    
    plt.figure(figsize=(6,4))
    plt.plot(ns, g, marker="o", markersize=3)
    plt.xlabel("n")
    plt.ylabel(r"$\|K^n\|_2$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("transient_growth_norm2.pdf", dpi=300)
    plt.close()


    # Gram matrices / diagnostics
    LMDA, num_rej = float(params[0]), float(params[1])
    H = M0[0].copy()
    for i in range(1, len(M0)):
        H += M0[i] * (LMDA ** (2 * i))

    v1 = np.percentile(H, 95)
    v0 = np.percentile(H, 5)
    save_gram_matrix(H[:-1, :-1], filename="G_sst.pdf", vmin=v0, vmax=v1)

    v1 = np.percentile(K, 95)
    v0 = np.percentile(K, 5)
    save_gram_matrix(K, filename="K_sst.pdf", title="Koopman Matrix Heatmap",
                     label=r"$\hat{K}_{ij}$", vmin=v0, vmax=v1)

    KKh = K @ K.conj().T
    save_gram_matrix(KKh, filename="KKh_sst.pdf", title="K K*",
                     label=r"$(\hat{K}\hat{K}^*)_{ij}$", vmin=0, vmax=1)

    KhK = K.conj().T @ K
    save_gram_matrix(KhK, filename="KhK_sst.pdf", title="K* K",
                     label=r"$(\hat{K}^*\hat{K})_{ij}$", vmin=0, vmax=1)

    # -----------------------------
    # non-normality (global)
    # -----------------------------
    denom = np.linalg.norm(KhK, ord="fro")
    diff = KhK - KKh
    eps_normal = np.linalg.norm(diff, ord="fro") / (denom + 1e-300)
    print(f"# non-normality ||K^*K - KK^*||_F / ||K^*K||_F = {eps_normal:.6e}")

    # -----------------------------
    # energy of off-diagonal in KhK (your original)
    # -----------------------------
    diag_part = np.diag(np.diag(KhK))
    offdiag_part = KhK - diag_part
    eps_offdiag = np.linalg.norm(offdiag_part, ord="fro") / (denom + 1e-300)
    print(f"# off-diagonal energy ratio (KhK) = {eps_offdiag:.6e}")

    # ============================================================
    # Added diagnostics: decomposition of non-normality
    # ============================================================
    # (A) Diagonal vs off-diagonal contribution in (KhK - KKh)
    diff_diag = np.diag(np.diag(diff))
    diff_off  = diff - diff_diag

    r_diag = np.linalg.norm(diff_diag, ord="fro") / (denom + 1e-300)
    r_off  = np.linalg.norm(diff_off,  ord="fro") / (denom + 1e-300)
    # sanity: eps_normal^2 ~ r_diag^2 + r_off^2 (up to roundoff)
    print(f"# non-normality split: diag-part/||KhK||_F = {r_diag:.6e}")
    print(f"# non-normality split: offdiag-part/||KhK||_F = {r_off:.6e}")
    print(f"# check: sqrt(diag^2+off^2) = {np.sqrt(r_diag*r_diag + r_off*r_off):.6e}")

    # (B) Compare diagonals directly:
    #     diag(KhK)_i = ||column_i(K)||^2, diag(KKh)_i = ||row_i(K)||^2
    d_col = np.real(np.diag(KhK))   # column norms^2
    d_row = np.real(np.diag(KKh))   # row norms^2
    rel_diag_gap = np.linalg.norm(d_col - d_row) / (np.linalg.norm(d_col) + 1e-300)
    print(f"# diag gap ||diag(K^*K)-diag(KK^*)||_2 / ||diag(K^*K)||_2 = {rel_diag_gap:.6e}")

    # (C) "hub" indices: large column/row norms (top-k)
    topk = 10
    idx_col = np.argsort(d_col)[::-1][:topk]
    idx_row = np.argsort(d_row)[::-1][:topk]
    print("# top column-norm^2 indices (diag(K^*K)):", idx_col.tolist())
    print("# top row-norm^2 indices    (diag(KK^*)):", idx_row.tolist())
    print("# top column-norm^2 values:", d_col[idx_col])
    print("# top row-norm^2 values   :", d_row[idx_row])

    # (D) Optional: cosine similarity between KhK and KKh (Frobenius inner product)
    #     if ~0 and norms similar => eps_normal ~ sqrt(2) * ||KhK||/||KhK|| = sqrt(2)
    inner = np.vdot(KhK, KKh).real
    cos_F = inner / ((np.linalg.norm(KhK, 'fro') * np.linalg.norm(KKh, 'fro')) + 1e-300)
    print(f"# Frobenius cosine sim <KhK,KKh>/(||KhK||_F||KKh||_F) = {cos_F:.6e}")

    # (E) Optional: off-diagonal energy of KKh as well (symmetry check)
    diag_KKh = np.diag(np.diag(KKh))
    off_KKh  = KKh - diag_KKh
    eps_offdiag_KKh = np.linalg.norm(off_KKh, ord="fro") / (np.linalg.norm(KKh, ord="fro") + 1e-300)
    print(f"# off-diagonal energy ratio (KKh) = {eps_offdiag_KKh:.6e}")
    absK = np.abs(K)
    imax = np.unravel_index(np.argmax(absK), K.shape)
    print("# argmax |K| =", imax, " value =", K[imax])

    froK = np.linalg.norm(K, 'fro')
    maxK = absK[imax]
    print("# max|K| / ||K||_F =", float(maxK / (froK + 1e-300)))

    col_norm2 = np.sum(absK**2, axis=0)  # (n,)
    row_norm2 = np.sum(absK**2, axis=1)  # (n,)
    print("# max col_norm2 idx,val =", int(np.argmax(col_norm2)), float(np.max(col_norm2)))
    print("# max row_norm2 idx,val =", int(np.argmax(row_norm2)), float(np.max(row_norm2)))

    
    # --- max column-energy ratio diagnostics ---
    # col_norm2[i] = ||K[:, i]||_2^2 = (K^* K)_{ii}
    col_norm2 = np.real(np.diag(KhK))  # (n,)
    
    # total energy: ||K||_F^2 = tr(K^*K)
    fro2 = float(np.real(np.trace(KhK)))
    
    if fro2 <= 0.0:
        print("# max column-energy ratio = NaN (||K||_F^2 <= 0)")
    else:
        i_max = int(np.argmax(col_norm2))
        ratio = float(col_norm2[i_max] / fro2)
        print(f"# max column-energy ratio = {ratio:.6e}")
        print(f"# argmax column-energy idx = {i_max}, value = {col_norm2[i_max]:.6e}, total = {fro2:.6e}")

    # optional: show top-k indices
    k = 10
    topk = np.argsort(col_norm2)[::-1][:k]
    print(f"# top{k} column-energy idx = {topk.tolist()}")
    print(f"# top{k} column-energy val = {col_norm2[topk]}")
