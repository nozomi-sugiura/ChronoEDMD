import numpy as np
from math import factorial
import os, sys
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


# ============================================================
# Parameters
# ============================================================

DATA_FILE = "data/l96.npz"
X_KEY = "X"
SEG_KEY = "seg"

ORDER = 3

# Explicit Sig-EDMD rank. For d=10, ORDER=3, dim = 1+10+100+1000 = 1111.
SIG_RANK = 1100

# SPK rank. None means all numerically effective directions.
SPK_RANK = None

# Annual-mean Markov baseline rank.
MEAN_RANK = None

# Kiraly-kEDMD rank. Use 1100 to match Sig.
KIRALY_RANK = 1100

RTOL = 1e-10
TRAIN_FRACTION = 0.8
N_BLOCK = 12
MAX_LEAD = 12
RIDGE = 1e-8

# Use only first NUSE blocks. Set NUSE=None for full data.
NUSE = None

# Kiraly Algorithm-3 padding. 0 is simplest and closest to explicit node path.
I_PAD = 0

# Parallel jobs for Kiraly Gram matrix. Use 4 first; increase if memory allows.
KIRALY_N_JOBS = 4

SAVE_KIRALY_KALL = True
LOAD_KIRALY_KALL_IF_EXISTS = True

OUT_DIR = "figs_sig_kiraly_spk_l96"
os.makedirs(OUT_DIR, exist_ok=True)

# Save per-lead prediction statistics for seed/model averaging.
SAVE_STATS = True
SAVE_STATS_CSV = True



# ============================================================
# Experiment metadata and stats saving
# ============================================================

def _npz_scalar(data, key, default=None):
    """Read a scalar metadata field from an npz file if present."""
    if key not in data:
        return default
    val = data[key]
    try:
        if np.ndim(val) == 0:
            return val.item()
        if np.size(val) == 1:
            return np.asarray(val).reshape(()).item()
    except Exception:
        pass
    return val


def _sanitize_tag_value(x):
    """Convert metadata value to a compact filename-safe string."""
    if x is None:
        return None
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="replace")
    if isinstance(x, float):
        text = f"{x:g}"
    else:
        text = str(x)
    text = text.strip()
    for ch in [" ", "/", "\\", ":", ";", ",", "=", "(", ")", "[", "]"]:
        text = text.replace(ch, "_")
    text = text.replace(".", "p")
    text = text.replace("-", "m")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def make_data_tag(data):
    """
    Construct a filename tag from data metadata.

    This is important when running multiple seeds: the Kiraly Gram matrix and
    stats files must not be accidentally reused across different simulated data.
    """
    model = _sanitize_tag_value(_npz_scalar(data, "MODEL", "l96"))

    parts = [model]
    for key, short in [
        ("SEED", "seed"),
        ("SEED_DYN", "sdyn"),
        ("SEED_OBS", "sobs"),
        ("K", "K"),
        ("J", "J"),
        ("K_FULL", "Kfull"),
        ("K_OBS", "Kobs"),
        ("F", "F"),
        ("F0", "F0"),
        ("h", "h"),
        ("c", "c"),
        ("b", "b"),
        ("FORCING_KIND", "forcing"),
        ("FORCING_AMP", "famp"),
        ("RANDOM_MIX", "rmix"),
        ("DT", "dt"),
        ("N_STEPS", "n"),
        ("SPINUP", "spin"),
    ]:
        val = _npz_scalar(data, key, None)
        sval = _sanitize_tag_value(val)
        if sval is not None and sval != "":
            parts.append(f"{short}{sval}")

    return "_".join(parts)


def save_prediction_stats(
    filename_npz,
    filename_csv,
    leads,
    sig_rmse, sig_kpc, sig_error,
    kir_rmse, kir_kpc, kir_error,
    spk_rmse, spk_kpc, spk_error,
    mean_rmse, mean_kpc, mean_error,
    metadata,
):
    """Save per-lead metrics in both npz and optional csv form."""
    leads = np.asarray(leads, dtype=int)

    arrays = dict(
        leads=leads,
        sigedmd_rmse=np.asarray(sig_rmse, dtype=np.float64),
        sigedmd_kpc=np.asarray(sig_kpc, dtype=np.float64),
        sigedmd_error=np.asarray(sig_error, dtype=np.float64),
        sig_rmse=np.asarray(kir_rmse, dtype=np.float64),
        sig_kpc=np.asarray(kir_kpc, dtype=np.float64),
        sig_error=np.asarray(kir_error, dtype=np.float64),
        spk_rmse=np.asarray(spk_rmse, dtype=np.float64),
        spk_kpc=np.asarray(spk_kpc, dtype=np.float64),
        spk_error=np.asarray(spk_error, dtype=np.float64),
        mean_rmse=np.asarray(mean_rmse, dtype=np.float64),
        mean_kpc=np.asarray(mean_kpc, dtype=np.float64),
        mean_error=np.asarray(mean_error, dtype=np.float64),
    )

    # Store metadata as scalar arrays/strings in the npz file.
    meta_arrays = {}
    for key, val in metadata.items():
        if val is None:
            continue
        try:
            meta_arrays[key] = np.asarray(val)
        except Exception:
            meta_arrays[key] = np.asarray(str(val))

    np.savez(filename_npz, **arrays, **meta_arrays)
    print("Saved stats to", filename_npz)

    if filename_csv is not None:
        header = (
            "lead,"
            "sig_rmse,sig_kpc,sig_error,"
            "spk_rmse,spk_kpc,spk_error,"
            "sigedmd_rmse,sigedmd_kpc,sigedmd_error,"
            "mean_rmse,mean_kpc,mean_error"
        )
        table = np.column_stack([
            leads,
            arrays["sig_rmse"], arrays["sig_kpc"], arrays["sig_error"],
            arrays["spk_rmse"], arrays["spk_kpc"], arrays["spk_error"],
            arrays["sigedmd_rmse"], arrays["sigedmd_kpc"], arrays["sigedmd_error"],
            arrays["mean_rmse"], arrays["mean_kpc"], arrays["mean_error"],
        ])
        np.savetxt(
            filename_csv,
            table,
            delimiter=",",
            header=header,
            comments="",
            fmt=["%d"] + ["%.16e"] * (table.shape[1] - 1),
        )
        print("Saved stats CSV to", filename_csv)


# ============================================================
# Explicit tensor signatures
# ============================================================

def tensor_power(v, k):
    if k == 0:
        return np.array(1.0)
    out = v
    for _ in range(k - 1):
        out = np.multiply.outer(out, v)
    return out


def tensor_exp_increment(dx, level):
    return [tensor_power(dx, k) / factorial(k) for k in range(level + 1)]


def tensor_concat(A, B):
    if A.ndim == 0:
        return A * B
    if B.ndim == 0:
        return B * A
    return np.multiply.outer(A, B)


def chen_product(S, T, level):
    out = []
    for k in range(level + 1):
        acc = None
        for i in range(k + 1):
            term = tensor_concat(S[i], T[k - i])
            acc = term if acc is None else acc + term
        out.append(acc)
    return out


def path_signature(path, level=3):
    path = np.asarray(path, dtype=np.float64)
    d = path.shape[1]

    S = [np.array(1.0)]
    for k in range(1, level + 1):
        S.append(np.zeros((d,) * k, dtype=np.float64))

    for dx in path[1:] - path[:-1]:
        E = tensor_exp_increment(dx, level)
        S = chen_product(S, E, level)

    return S


def flatten_signature(S):
    return np.concatenate([np.ravel(A) for A in S])


def compute_signature_features(segments, level=3):
    Phi = []
    for i, path in enumerate(segments):
        Phi.append(flatten_signature(path_signature(path, level=level)))
        if (i + 1) % 500 == 0:
            print(f"computed explicit signatures: {i + 1}")
    return np.vstack(Phi)


# ============================================================
# Generic reduced EDMD in explicit feature space
# ============================================================

def fit_edmd_reduced(Phi_X, Phi_Y, rtol=1e-10, rank=None):
    """
    Row-vector convention:
        Phi_Y approx Phi_X @ K_full.

    Reduced coordinate:
        Z = Phi @ V_r.
    """
    U, s, Vt = np.linalg.svd(Phi_X, full_matrices=False)

    tol = rtol * s[0]
    idx = np.where(s > tol)[0]
    if rank is not None:
        idx = idx[:rank]

    U_r = U[:, idx]
    s_r = s[idx]
    V_r = Vt[idx, :].T

    K_r = (U_r.T @ Phi_Y @ V_r) / s_r[:, None]
    return K_r, V_r, s_r


def fit_linear_decoder(Z_train, Y_train, ridge=1e-8):
    Z = np.asarray(Z_train, dtype=np.float64)
    Y = np.asarray(Y_train, dtype=np.float64)
    r = Z.shape[1]
    lhs = Z.T @ Z + ridge * np.eye(r)
    rhs = Z.T @ Y
    return np.linalg.solve(lhs, rhs)


# ============================================================
# SPK explicit feature
# ============================================================

def compute_spk_features_from_cumulative_paths(segments):
    """
    SPK aligned-increment linear feature on cumulative paths.
    Since increments of C_i are the original 12-value sequence,
    this is simply the flattened increment sequence.
    """
    increments = segments[:, 1:, :] - segments[:, :-1, :]
    return increments.reshape(segments.shape[0], -1)
def compute_annual_mean_features_from_cumulative_paths(segments):
    """
    Annual-mean Markov feature from cumulative paths.

    segments: (n_seg, 13, d)
        cumulative paths whose increments are the original 12 values.

    Returns
    -------
    Phi_mean: (n_seg, d)
        annual mean vector for each segment.
    """
    increments = segments[:, 1:, :] - segments[:, :-1, :]  # (n_seg, 12, d)
    annual_mean = np.mean(increments, axis=1)              # (n_seg, d)
    return annual_mean


# ============================================================
# Kiraly--Oberhauser Algorithm-3 signature kernel
# ============================================================

def linear_kernel(x, y):
    return float(np.dot(x, y))


def padding(X, p):
    X = np.asarray(X, dtype=np.float64)
    if p == 0:
        return X.copy()

    n, d = X.shape
    lx = n - 1
    X2 = np.zeros(((p + 1) * lx + 1, d), dtype=X.dtype)
    X2[0, :] = X[0, :]

    for i in range(lx):
        for j in range(p + 1):
            X2[(p + 1) * i + 1 + j, :] = (
                ((j + 1) * X[i + 1, :] + (p - j) * X[i, :]) / (p + 1)
            )
    return X2


def suffix_sum_strict(A):
    S = np.cumsum(np.cumsum(A[::-1, ::-1], axis=0), axis=1)[::-1, ::-1]
    out = np.zeros_like(A)
    out[:-1, :-1] = S[1:, 1:]
    return out


def signature_kernel_alg3(X, Y, m=3):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    nx = X.shape[0]
    ny = Y.shape[0]

    kij = np.empty((nx, ny), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            kij[i, j] = linear_kernel(X[i], Y[j])

    C = kij[1:, 1:] + kij[:-1, :-1] - kij[:-1, 1:] - kij[1:, :-1]

    A = C.copy()
    for _ in range(2, m + 1):
        Q = suffix_sum_strict(A)
        A = C * (1.0 + Q)

    return 1.0 + A.sum()


def _compute_kiraly_row(i, seg_pad, order):
    row_vals = np.empty(i + 1, dtype=np.float64)
    for j in range(i + 1):
        row_vals[j] = signature_kernel_alg3(seg_pad[i], seg_pad[j], m=order)
    return i, row_vals


def build_kiraly_Kall_parallel(segments, order=3, i_pad=0, n_jobs=4):
    N = len(segments)
    print(
        f"Building Kiraly K_all in parallel: N={N}, order={order}, "
        f"i_pad={i_pad}, n_jobs={n_jobs}"
    )

    seg_pad = [padding(path, i_pad) for path in segments]

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_compute_kiraly_row)(i, seg_pad, order)
        for i in range(N)
    )

    K_all = np.empty((N, N), dtype=np.float64)
    for i, row_vals in results:
        K_all[i, : i + 1] = row_vals
        K_all[: i + 1, i] = row_vals

    return K_all


# ============================================================
# Williams-type kernel EDMD
# ============================================================

def fit_kedmd_from_kernel_matrix(G, A, rtol=1e-10, rank=None):
    """
    G_ab = k(x_a, x_b)
    A_ab = k(y_a, x_b), where y_a is the next state of x_a.
    """
    G = 0.5 * (G + G.T)

    evals, Q = np.linalg.eigh(G)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    Q = Q[:, idx]

    tol = rtol * evals[0]
    keep_idx = np.where(evals > tol)[0]
    if rank is not None:
        keep_idx = keep_idx[:rank]

    evals_r = evals[keep_idx]
    Q_r = Q[:, keep_idx]
    Sigma = np.sqrt(evals_r)

    Khat = ((1.0 / Sigma)[:, None] * (Q_r.T @ A @ Q_r)) * (1.0 / Sigma)[None, :]
    return Khat, Q_r, Sigma, evals_r


def kernel_coordinates_from_Kall(K_all, Q_r, Sigma, sample_idx, train_idx):
    K_sample_train = K_all[np.ix_(sample_idx, train_idx)]
    return K_sample_train @ (Q_r / Sigma[None, :])


# ============================================================
# Physical target and metrics
# ============================================================

def make_block_values_from_X(X, block_indices, n_block=12):
    Y = []
    for i in block_indices:
        start = n_block * int(i)
        block = X[start : start + n_block]
        if block.shape[0] != n_block:
            raise ValueError("X is too short for requested block.")
        Y.append(block)
    Y = np.asarray(Y, dtype=np.float64)
    return Y.reshape(len(block_indices), -1)


def abs_rmse(Y_true, Y_pred):
    return np.sqrt(np.mean((Y_true - Y_pred) ** 2))


def values_to_cumulative_paths(Y, n_block, d):
    Y = np.asarray(Y, dtype=np.float64)
    N = Y.shape[0]
    values = Y.reshape(N, n_block, d)
    paths = np.zeros((N, n_block + 1, d), dtype=np.float64)
    paths[:, 1:, :] = np.cumsum(values, axis=1)
    return paths


def signature_kernel_cosine_from_values(Y_pred, Y_true, n_block, d, level=3, eps=1e-14):
    C_pred = values_to_cumulative_paths(Y_pred, n_block=n_block, d=d)
    C_true = values_to_cumulative_paths(Y_true, n_block=n_block, d=d)

    kpc_each = np.empty(C_pred.shape[0], dtype=np.float64)

    for i in range(C_pred.shape[0]):
        S_pred = flatten_signature(path_signature(C_pred[i], level=level))
        S_true = flatten_signature(path_signature(C_true[i], level=level))

        k_pt = float(np.dot(S_pred, S_true))
        k_pp = float(np.dot(S_pred, S_pred))
        k_tt = float(np.dot(S_true, S_true))

        kpc_each[i] = k_pt / (np.sqrt(k_pp * k_tt) + eps)

    return float(np.mean(kpc_each)), kpc_each


def signature_error_from_kpc(kpc):
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - kpc))))


def evaluate_prediction(Y_true, Y_pred, n_block, d, level):
    rmse = abs_rmse(Y_true, Y_pred)
    kpc, _ = signature_kernel_cosine_from_values(
        Y_pred,
        Y_true,
        n_block=n_block,
        d=d,
        level=level,
    )
    sig_error = signature_error_from_kpc(kpc)
    return rmse, kpc, sig_error


# ============================================================
# Plotting
# ============================================================

def plot_skill_curve4(
    leads,
    sig_vals,
    kir_vals,
    spk_vals,
    mean_vals,
    ylabel,
    filename,
):
    plt.figure(figsize=(6.4, 4.2))

    # SIG: Kiraly signature kernel
    plt.plot(
        leads, kir_vals,
        color="tab:blue",
        marker="o",
        linestyle="-",
        linewidth=2.5,
        markersize=6,
        label="SIG",
    )

    # SPK: aligned increment baseline
    plt.plot(
        leads, spk_vals,
        color="tab:orange",
        marker="^",
        linestyle="-",
        linewidth=2.2,
        markersize=6,
        label="SPK",
    )

    # Explicit Sig-EDMD
    plt.plot(
        leads, sig_vals,
        color="tab:green",
        marker="s",
        linestyle="--",
        linewidth=2.2,
        markersize=6,
        label="Sig-EDMD",
    )

    # Annual Mean Markov baseline
    plt.plot(
        leads, mean_vals,
        color="tab:red",
        marker="d",
        linestyle=":",
        linewidth=2.2,
        markersize=6,
        label="Mean",
    )

    plt.xlabel("Lead time")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

def explicit_signature_inner_product(X, Y, level=3):
    SX = flatten_signature(path_signature(X, level=level))
    SY = flatten_signature(path_signature(Y, level=level))
    return float(np.dot(SX, SY))


def check_alg3_vs_explicit_signature(segments, order=3, pairs=None):
    if pairs is None:
        pairs = [
            (0, 0),
            (0, 1),
            (1, 1),
            (10, 11),
            (30, 29),
            (100, 101),
        ]

    print("")
    print("============================================================")
    print("Check: Algorithm-3 kernel vs explicit <Sig,Sig>")
    print("============================================================")
    print("order =", order)

    rel_diffs = []

    for i, j in pairs:
        if i >= len(segments) or j >= len(segments):
            continue

        k_exp = explicit_signature_inner_product(
            segments[i],
            segments[j],
            level=order,
        )

        k_alg = signature_kernel_alg3(
            segments[i],
            segments[j],
            m=order,
        )

        diff = k_alg - k_exp
        rel = abs(diff) / (abs(k_exp) + 1e-300)
        rel_diffs.append(rel)

        print(f"pair ({i}, {j})")
        print(f"  explicit <Sig,Sig> = {k_exp:.16e}")
        print(f"  alg3 kernel        = {k_alg:.16e}")
        print(f"  diff               = {diff:.16e}")
        print(f"  rel diff           = {rel:.16e}")

    if len(rel_diffs) > 0:
        print("")
        print("summary")
        print(f"  max rel diff  = {np.max(rel_diffs):.16e}")
        print(f"  mean rel diff = {np.mean(rel_diffs):.16e}")

    print("============================================================")
    print("")
    
# ============================================================
# Main
# ============================================================

def main():
    data = np.load(DATA_FILE)
    X = data[X_KEY]
    segments = data[SEG_KEY]

    data_tag = make_data_tag(data)
    print("DATA_FILE:", DATA_FILE)
    print("DATA_TAG:", data_tag)

    if NUSE is not None:
        X = X[: NUSE * N_BLOCK]
        segments = segments[:NUSE]

    print("X shape:", X.shape)
    print("segments shape:", segments.shape)

    if "SEG_SCALE" in data:
        print("SEG_SCALE:", float(data["SEG_SCALE"]))

    if "SEG_SCALE_KIND" in data:
        print("SEG_SCALE_KIND:", str(data["SEG_SCALE_KIND"]))

    check_alg3_vs_explicit_signature(
        segments,
        order=ORDER,
    )
# sys.exit()
    n_seg, n_nodes, d = segments.shape
    if n_nodes != N_BLOCK + 1:
        raise ValueError("seg must have 13 nodes for N_BLOCK=12.")

    max_start_from_X = X.shape[0] // N_BLOCK
    if n_seg > max_start_from_X:
        raise ValueError("Number of segments is inconsistent with X length.")

    n_transitions = n_seg - 1
    n_train = int(TRAIN_FRACTION * n_transitions)
    train_idx = np.arange(n_train)
    train_next_idx = train_idx + 1

    Y_train = make_block_values_from_X(X, train_idx, n_block=N_BLOCK)

    # --------------------------------------------------------
    # 1. Explicit Sig-EDMD
    # --------------------------------------------------------
    Phi_sig = compute_signature_features(segments, level=ORDER)
    print("Phi_sig shape:", Phi_sig.shape)

    K_sig, V_sig, _ = fit_edmd_reduced(
        Phi_sig[train_idx],
        Phi_sig[train_next_idx],
        rtol=RTOL,
        rank=SIG_RANK,
    )
    print("Sig reduced rank:", K_sig.shape[0])

    Z_sig_train = Phi_sig[train_idx] @ V_sig
    W_sig = fit_linear_decoder(Z_sig_train, Y_train, ridge=RIDGE)

    # --------------------------------------------------------
    # 2. SPK-kEDMD, explicit aligned-increment feature
    # --------------------------------------------------------
    Phi_spk = compute_spk_features_from_cumulative_paths(segments)
    print("Phi_spk shape:", Phi_spk.shape)

    K_spk, V_spk, _ = fit_edmd_reduced(
        Phi_spk[train_idx],
        Phi_spk[train_next_idx],
        rtol=RTOL,
        rank=SPK_RANK,
    )
    print("SPK reduced rank:", K_spk.shape[0])

    Z_spk_train = Phi_spk[train_idx] @ V_spk
    W_spk = fit_linear_decoder(Z_spk_train, Y_train, ridge=RIDGE)

    # --------------------------------------------------------
    # 2b. Annual-mean Markov baseline
    # --------------------------------------------------------
    Phi_mean = compute_annual_mean_features_from_cumulative_paths(segments)
    print("Phi_mean shape:", Phi_mean.shape)

    K_mean, V_mean, _ = fit_edmd_reduced(
        Phi_mean[train_idx],
        Phi_mean[train_next_idx],
        rtol=RTOL,
        rank=MEAN_RANK,
    )
    print("Annual Mean reduced rank:", K_mean.shape[0])

    Z_mean_train = Phi_mean[train_idx] @ V_mean
    W_mean = fit_linear_decoder(Z_mean_train, Y_train, ridge=RIDGE)

    # --------------------------------------------------------
    # 3. Kiraly-kEDMD
    # --------------------------------------------------------
    kiraly_file = os.path.join(
        OUT_DIR,
        f"Kall_kiraly_{data_tag}_N{n_seg}_m{ORDER}_pad{I_PAD}.npz",
    )

    if LOAD_KIRALY_KALL_IF_EXISTS and os.path.exists(kiraly_file):
        print("Loading Kiraly K_all from", kiraly_file)
        Kall_kiraly = np.load(kiraly_file)["K_all"]
    else:
        Kall_kiraly = build_kiraly_Kall_parallel(
            segments,
            order=ORDER,
            i_pad=I_PAD,
            n_jobs=KIRALY_N_JOBS,
        )
        if SAVE_KIRALY_KALL:
            np.savez(kiraly_file, K_all=Kall_kiraly)
            print("Saved Kiraly K_all to", kiraly_file)

    G_kir = Kall_kiraly[np.ix_(train_idx, train_idx)]
    A_kir = Kall_kiraly[np.ix_(train_next_idx, train_idx)]

    K_kir, Q_kir, Sigma_kir, _ = fit_kedmd_from_kernel_matrix(
        G_kir,
        A_kir,
        rtol=RTOL,
        rank=KIRALY_RANK,
    )
    print("Kiraly reduced rank:", K_kir.shape[0])

    Z_kir_train = kernel_coordinates_from_Kall(
        Kall_kiraly,
        Q_kir,
        Sigma_kir,
        train_idx,
        train_idx,
    )
    W_kir = fit_linear_decoder(Z_kir_train, Y_train, ridge=RIDGE)

    # --------------------------------------------------------
    # Multi-lead prediction
    # --------------------------------------------------------
    print(
        "lead "
        "Sig_RMSE Sig_kPC Sig_error "
        "Kiraly_RMSE Kiraly_kPC Kiraly_error "
        "SPK_RMSE SPK_kPC SPK_error "
        "Mean_RMSE Mean_kPC Mean_error"
    )

    leads_list = []
    sig_rmse_list, sig_kpc_list, sig_err_list = [], [], []
    kir_rmse_list, kir_kpc_list, kir_err_list = [], [], []
    spk_rmse_list, spk_kpc_list, spk_err_list = [], [], []
    mean_rmse_list, mean_kpc_list, mean_err_list = [], [], []

    for lead in range(1, MAX_LEAD + 1):
        test_idx = np.arange(n_train, n_seg - lead)
        if len(test_idx) == 0:
            break

        future_idx = test_idx + lead
        Y_true = make_block_values_from_X(X, future_idx, n_block=N_BLOCK)

        # Sig
        Z_sig_test = Phi_sig[test_idx] @ V_sig
        Z_sig_pred = Z_sig_test @ np.linalg.matrix_power(K_sig, lead)
        Y_sig_pred = Z_sig_pred @ W_sig
        sig_rmse, sig_kpc, sig_error = evaluate_prediction(
            Y_true, Y_sig_pred, n_block=N_BLOCK, d=d, level=ORDER
        )

        # Kiraly
        Z_kir_test = kernel_coordinates_from_Kall(
            Kall_kiraly, Q_kir, Sigma_kir, test_idx, train_idx
        )
        Z_kir_pred = Z_kir_test @ np.linalg.matrix_power(K_kir, lead)
        Y_kir_pred = Z_kir_pred @ W_kir
        kir_rmse, kir_kpc, kir_error = evaluate_prediction(
            Y_true, Y_kir_pred, n_block=N_BLOCK, d=d, level=ORDER
        )

        # SPK
        Z_spk_test = Phi_spk[test_idx] @ V_spk
        Z_spk_pred = Z_spk_test @ np.linalg.matrix_power(K_spk, lead)
        Y_spk_pred = Z_spk_pred @ W_spk
        spk_rmse, spk_kpc, spk_error = evaluate_prediction(
            Y_true, Y_spk_pred, n_block=N_BLOCK, d=d, level=ORDER
        )
        # Annual Mean
        Z_mean_test = Phi_mean[test_idx] @ V_mean
        Z_mean_pred = Z_mean_test @ np.linalg.matrix_power(K_mean, lead)
        Y_mean_pred = Z_mean_pred @ W_mean
        mean_rmse, mean_kpc, mean_error = evaluate_prediction(
            Y_true, Y_mean_pred, n_block=N_BLOCK, d=d, level=ORDER
        )
        print(
            f"{lead:2d} "
            f"{sig_rmse:.10f} {sig_kpc:.10f} {sig_error:.10f} "
            f"{kir_rmse:.10f} {kir_kpc:.10f} {kir_error:.10f} "
            f"{spk_rmse:.10f} {spk_kpc:.10f} {spk_error:.10f} "
            f"{mean_rmse:.10f} {mean_kpc:.10f} {mean_error:.10f}"
        )

        leads_list.append(lead)

        sig_rmse_list.append(sig_rmse)
        sig_kpc_list.append(sig_kpc)
        sig_err_list.append(sig_error)

        kir_rmse_list.append(kir_rmse)
        kir_kpc_list.append(kir_kpc)
        kir_err_list.append(kir_error)

        spk_rmse_list.append(spk_rmse)
        spk_kpc_list.append(spk_kpc)
        spk_err_list.append(spk_error)

        mean_rmse_list.append(mean_rmse)
        mean_kpc_list.append(mean_kpc)
        mean_err_list.append(mean_error)

    plot_skill_curve4(
        leads_list,
        sig_rmse_list,
        kir_rmse_list,
        spk_rmse_list,
        mean_rmse_list,
        ylabel="RMSE",
        filename=os.path.join(OUT_DIR, "rmse_compare_sig_spk_sigedmd_mean.pdf"),
    )

    plot_skill_curve4(
        leads_list,
        sig_kpc_list,
        kir_kpc_list,
        spk_kpc_list,
        mean_kpc_list,
        ylabel="kPC",
        filename=os.path.join(OUT_DIR, "kpc_compare_sig_spk_sigedmd_mean.pdf"),
    )

    plot_skill_curve4(
        leads_list,
        sig_err_list,
        kir_err_list,
        spk_err_list,
        mean_err_list,
        ylabel="signature error",
        filename=os.path.join(OUT_DIR, "sigerr_compare_sig_spk_sigedmd_mean.pdf"),
    )

    if SAVE_STATS:
        stats_base = (
            f"stats_{data_tag}_order{ORDER}_"
            f"sigrank{SIG_RANK}_kirrank{KIRALY_RANK}_"
            f"ntrain{n_train}_nseg{n_seg}"
        )
        stats_npz = os.path.join(OUT_DIR, stats_base + ".npz")
        stats_csv = os.path.join(OUT_DIR, stats_base + ".csv") if SAVE_STATS_CSV else None

        metadata = {
            "data_file": DATA_FILE,
            "data_tag": data_tag,
            "order": ORDER,
            "sig_rank": SIG_RANK,
            "spk_rank": -1 if SPK_RANK is None else SPK_RANK,
            "mean_rank": -1 if MEAN_RANK is None else MEAN_RANK,
            "kiraly_rank": KIRALY_RANK,
            "rtol": RTOL,
            "train_fraction": TRAIN_FRACTION,
            "n_block": N_BLOCK,
            "max_lead": MAX_LEAD,
            "ridge": RIDGE,
            "nuse": -1 if NUSE is None else NUSE,
            "i_pad": I_PAD,
            "n_seg": n_seg,
            "n_train": n_train,
            "d": d,
            "seg_scale": _npz_scalar(data, "SEG_SCALE", np.nan),
            "seg_scale_kind": _npz_scalar(data, "SEG_SCALE_KIND", ""),
            "model": _npz_scalar(data, "MODEL", ""),
            "seed": _npz_scalar(data, "SEED", np.nan),
            "seed_dyn": _npz_scalar(data, "SEED_DYN", np.nan),
            "K": _npz_scalar(data, "K", np.nan),
            "J": _npz_scalar(data, "J", np.nan),
            "F": _npz_scalar(data, "F", np.nan),
            "h": _npz_scalar(data, "h", np.nan),
            "c": _npz_scalar(data, "c", np.nan),
            "b": _npz_scalar(data, "b", np.nan),
            "DT": _npz_scalar(data, "DT", np.nan),
            "N_STEPS": _npz_scalar(data, "N_STEPS", np.nan),
            "SPINUP": _npz_scalar(data, "SPINUP", np.nan),
        }

        save_prediction_stats(
            stats_npz,
            stats_csv,
            leads_list,
            sig_rmse_list, sig_kpc_list, sig_err_list,
            kir_rmse_list, kir_kpc_list, kir_err_list,
            spk_rmse_list, spk_kpc_list, spk_err_list,
            mean_rmse_list, mean_kpc_list, mean_err_list,
            metadata,
        )

    print("Saved figures to", OUT_DIR)


if __name__ == "__main__":
    main()
