import numpy as np
import os, sys
import matplotlib.pyplot as plt
from math import factorial


# ============================================================
# Parameters
# ============================================================

DATA_FILE = "data/l96.npz"
X_KEY = "X"
SEG_KEY = "seg"

ORDER = 3

TRAIN_FRACTION = 0.8
N_BLOCK = 12
MAX_LEAD = 12
RIDGE = 1e-8

# Use only first NUSE blocks. Set NUSE=None for full data.
NUSE = None

OUT_DIR = "figs_snapshot_kedmd_l96"
os.makedirs(OUT_DIR, exist_ok=True)

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
    snap_rmse, snap_kpc, snap_error,
    metadata,
):
    """Save per-lead metrics in both npz and optional csv form."""
    leads = np.asarray(leads, dtype=int)

    arrays = dict(
        leads=leads,
        snap_rmse=np.asarray(snap_rmse, dtype=np.float64),
        snap_kpc=np.asarray(snap_kpc, dtype=np.float64),
        snap_error=np.asarray(snap_error, dtype=np.float64),
    )

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
        header = "lead,snap_rmse,snap_kpc,snap_error"
        rows = []
        for i, lead in enumerate(leads):
            rows.append(
                [
                    int(lead),
                    float(arrays["snap_rmse"][i]),
                    float(arrays["snap_kpc"][i]),
                    float(arrays["snap_error"][i]),
                ]
            )
        np.savetxt(
            filename_csv,
            np.asarray(rows, dtype=object),
            delimiter=",",
            header=header,
            comments="",
            fmt=["%d", "%.12g", "%.12g", "%.12g"],
        )
        print("Saved stats to", filename_csv)


# ============================================================
# Explicit signatures for kPC evaluation
# ============================================================

def tensor_power(v, m):
    if m == 0:
        return np.array([1.0], dtype=np.float64)
    out = np.array([1.0], dtype=np.float64)
    for _ in range(m):
        out = np.kron(out, v)
    return out


def tensor_exp_increment(v, level):
    out = []
    for m in range(level + 1):
        out.append(tensor_power(v, m) / factorial(m))
    return out


def tensor_concat(a, b):
    level = len(a) - 1
    out = []
    for m in range(level + 1):
        s = None
        for k in range(m + 1):
            term = np.kron(a[k], b[m - k])
            s = term if s is None else s + term
        out.append(s)
    return out


def path_signature(path, level):
    """
    Signature of a piecewise-linear path up to a given level.
    """
    path = np.asarray(path, dtype=np.float64)
    sig = [np.array([1.0], dtype=np.float64)]
    d = path.shape[1]
    for m in range(1, level + 1):
        sig.append(np.zeros(d ** m, dtype=np.float64))

    for n in range(path.shape[0] - 1):
        v = path[n + 1] - path[n]
        inc = tensor_exp_increment(v, level)
        sig = tensor_concat(sig, inc)

    return sig


def flatten_signature(sig):
    return np.concatenate(sig)


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
    return np.sqrt(max(2.0 * (1.0 - kpc), 0.0))


def abs_rmse(Y_true, Y_pred):
    return float(np.sqrt(np.mean((Y_true - Y_pred) ** 2)))


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
# Snapshot EDMD
# ============================================================

def fit_snapshot_edmd_linear(X, ridge=1e-8):
    """
    Fit a one-step linear EDMD model with Euclidean/linear observables.

    Row-vector convention:
        X_{n+1} approx X_n @ A.

    Parameters
    ----------
    X : ndarray, shape (T_train, d)
        Training snapshots.
    ridge : float
        Ridge regularization.

    Returns
    -------
    A : ndarray, shape (d, d)
        One-step snapshot Koopman matrix in row-vector convention.
    """
    X0 = np.asarray(X[:-1], dtype=np.float64)
    X1 = np.asarray(X[1:], dtype=np.float64)

    d = X0.shape[1]
    lhs = X0.T @ X0 + ridge * np.eye(d)
    rhs = X0.T @ X1
    A = np.linalg.solve(lhs, rhs)
    return A


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


def predict_target_blocks_from_tail(X, A_powers, block_indices, lead, n_block=12):
    """
    Predict target block i+lead from the tail snapshot of input block i.

    Block convention:
        x_{i,j} = X[12*i + j - 1],  j=1,...,12.

    The input snapshot is x_{i,12} = X[12*i + 11].
    The target snapshots are x_{i+lead,1},...,x_{i+lead,12}.

    Therefore the required one-step powers are
        12*lead - 11, ..., 12*lead.
    """
    preds = []
    d = X.shape[1]

    for i in block_indices:
        i = int(i)
        tail_index = n_block * i + (n_block - 1)
        x0 = X[tail_index]

        block_pred = np.empty((n_block, d), dtype=np.float64)
        for j0 in range(n_block):
            power = n_block * lead - (n_block - 1) + j0
            if power <= 0:
                raise ValueError("Prediction power must be positive.")
            block_pred[j0] = x0 @ A_powers[power]

        preds.append(block_pred)

    preds = np.asarray(preds, dtype=np.float64)
    return preds.reshape(len(block_indices), -1)


def precompute_powers(A, max_power):
    d = A.shape[0]
    powers = [np.eye(d, dtype=np.float64)]
    for _ in range(max_power):
        powers.append(powers[-1] @ A)
    return powers


# ============================================================
# Plotting
# ============================================================

def plot_skill_curve(leads, y, ylabel, filename):
    plt.figure(figsize=(6, 4))
    plt.plot(leads, y, marker="o", linewidth=2.0, label="Snapshot EDMD")
    plt.xlabel("lead (blocks)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", filename)


# ============================================================
# Main
# ============================================================

def main():
    data = np.load(DATA_FILE)
    X = np.asarray(data[X_KEY], dtype=np.float64)
    segments = data[SEG_KEY]

    data_tag = make_data_tag(data)
    print("DATA_FILE:", DATA_FILE)
    print("DATA_TAG:", data_tag)

    if NUSE is not None:
        X = X[: NUSE * N_BLOCK]
        segments = segments[:NUSE]

    print("X shape:", X.shape)
    print("segments shape:", segments.shape)

    n_seg, n_nodes, d = segments.shape
    if n_nodes != N_BLOCK + 1:
        raise ValueError("seg must have N_BLOCK+1 nodes.")

    max_start_from_X = X.shape[0] // N_BLOCK
    if n_seg > max_start_from_X:
        raise ValueError("Number of segments is inconsistent with X length.")

    n_transitions = n_seg - 1
    n_train = int(TRAIN_FRACTION * n_transitions)

    # Existing path-based experiments use block transitions
    # train_idx -> train_idx + 1 for train_idx=0,...,n_train-1.
    # Hence the training period includes blocks 0,...,n_train.
    train_time_end = (n_train + 1) * N_BLOCK
    if train_time_end > X.shape[0]:
        raise ValueError("Training time end exceeds X length.")

    X_train_snap = X[:train_time_end]
    print("n_seg:", n_seg)
    print("n_train block transitions:", n_train)
    print("snapshot training snapshots:", X_train_snap.shape[0])
    print("snapshot training one-step transitions:", X_train_snap.shape[0] - 1)

    A = fit_snapshot_edmd_linear(X_train_snap, ridge=RIDGE)
    print("A shape:", A.shape)

    A_powers = precompute_powers(A, max_power=N_BLOCK * MAX_LEAD)

    print("lead Snapshot_RMSE Snapshot_kPC Snapshot_error")
    leads_list = []
    snap_rmse_list, snap_kpc_list, snap_err_list = [], [], []

    for lead in range(1, MAX_LEAD + 1):
        test_idx = np.arange(n_train, n_seg - lead)
        if len(test_idx) == 0:
            break

        future_idx = test_idx + lead
        Y_true = make_block_values_from_X(X, future_idx, n_block=N_BLOCK)
        Y_pred = predict_target_blocks_from_tail(
            X,
            A_powers,
            test_idx,
            lead=lead,
            n_block=N_BLOCK,
        )

        rmse, kpc, sig_error = evaluate_prediction(
            Y_true, Y_pred, n_block=N_BLOCK, d=d, level=ORDER
        )

        print(f"{lead:2d} {rmse:.10f} {kpc:.10f} {sig_error:.10f}")

        leads_list.append(lead)
        snap_rmse_list.append(rmse)
        snap_kpc_list.append(kpc)
        snap_err_list.append(sig_error)

    plot_skill_curve(
        leads_list,
        snap_rmse_list,
        ylabel="RMSE",
        filename=os.path.join(OUT_DIR, "rmse_snapshot_edmd.pdf"),
    )
    plot_skill_curve(
        leads_list,
        snap_kpc_list,
        ylabel="kPC",
        filename=os.path.join(OUT_DIR, "kpc_snapshot_edmd.pdf"),
    )
    plot_skill_curve(
        leads_list,
        snap_err_list,
        ylabel="signature error",
        filename=os.path.join(OUT_DIR, "sigerr_snapshot_edmd.pdf"),
    )

    if SAVE_STATS:
        stats_base = f"stats_{data_tag}_snapshot_edmd_ntrain{n_train}_nseg{n_seg}"
        stats_npz = os.path.join(OUT_DIR, stats_base + ".npz")
        stats_csv = os.path.join(OUT_DIR, stats_base + ".csv") if SAVE_STATS_CSV else None

        metadata = {
            "data_file": DATA_FILE,
            "data_tag": data_tag,
            "order": ORDER,
            "train_fraction": TRAIN_FRACTION,
            "n_block": N_BLOCK,
            "max_lead": MAX_LEAD,
            "ridge": RIDGE,
            "nuse": -1 if NUSE is None else NUSE,
            "n_seg": n_seg,
            "n_train": n_train,
            "d": d,
            "method": "snapshot_edmd_linear_tail_start",
            "start_snapshot": "tail_of_input_block",
            "model": _npz_scalar(data, "MODEL", ""),
            "seed": _npz_scalar(data, "SEED", np.nan),
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
            snap_rmse_list,
            snap_kpc_list,
            snap_err_list,
            metadata,
        )

    print("Saved figures to", OUT_DIR)


if __name__ == "__main__":
    main()
