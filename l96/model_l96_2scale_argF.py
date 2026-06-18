import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


# ============================================================
# Two-scale Lorenz-96
# ============================================================

def l96_2scale_rhs(state, K=10, J=8, F=8.0, h=1.0, c=10.0, b=10.0):
    """
    Two-scale Lorenz-96 model.

    Slow variables:
        dX_k/dt =
            (X_{k+1} - X_{k-2}) X_{k-1}
            - X_k + F
            - (h c / b) sum_j Y_{j,k}

    Fast variables:
        dY_{j,k}/dt =
            c b (Y_{j-1,k} - Y_{j+2,k}) Y_{j+1,k}
            - c Y_{j,k}
            + (h c / b) X_k

    Periodic boundary conditions are used for k and j.
    """
    state = np.asarray(state, dtype=np.float64)

    X = state[:K]
    Y = state[K:].reshape(K, J)  # Y[k, j]

    # Slow equation
    dX = (
        (np.roll(X, -1) - np.roll(X, 2)) * np.roll(X, 1)
        - X
        + F
        - (h * c / b) * np.sum(Y, axis=1)
    )

    # Fast equation, periodic in j for each k
    dY = (
        c * b * (np.roll(Y, 1, axis=1) - np.roll(Y, -2, axis=1)) * np.roll(Y, -1, axis=1)
        - c * Y
        + (h * c / b) * X[:, None]
    )

    return np.concatenate([dX, dY.ravel()])


def rk4_step(f, x, dt, **kwargs):
    """
    One step of classical fourth-order Runge-Kutta.
    """
    k1 = f(x, **kwargs)
    k2 = f(x + 0.5 * dt * k1, **kwargs)
    k3 = f(x + 0.5 * dt * k2, **kwargs)
    k4 = f(x + dt * k3, **kwargs)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_l96_2scale(
    K=10,
    J=8,
    F=8.0,
    h=1.0,
    c=10.0,
    b=10.0,
    dt=0.005,
    n_steps=100000,
    spinup=25000,
    seed=1,
):
    """
    Simulate two-scale Lorenz-96 and return only the slow variables X.

    Returns
    -------
    X_out : ndarray, shape (n_steps, K)
        Slow variables.
    """
    rng = np.random.default_rng(seed)

    # Standard initialization near slow equilibrium, with small fast perturbations.
    X = F * np.ones(K, dtype=np.float64)
    X += 0.01 * rng.standard_normal(K)

    Y = 0.01 * rng.standard_normal((K, J))

    state = np.concatenate([X, Y.ravel()])

    # Spin-up
    for n in range(spinup):
        state = rk4_step(
            l96_2scale_rhs,
            state,
            dt,
            K=K,
            J=J,
            F=F,
            h=h,
            c=c,
            b=b,
        )

        if not np.all(np.isfinite(state)):
            raise FloatingPointError(f"Non-finite state during spinup at step {n}")

    # Main integration
    X_out = np.empty((n_steps, K), dtype=np.float64)

    for n in range(n_steps):
        state = rk4_step(
            l96_2scale_rhs,
            state,
            dt,
            K=K,
            J=J,
            F=F,
            h=h,
            c=c,
            b=b,
        )

        if not np.all(np.isfinite(state)):
            raise FloatingPointError(f"Non-finite state during main run at step {n}")

        X_out[n] = state[:K]

    return X_out


# ============================================================
# Segment construction and scale
# ============================================================

def make_segments_cumulative_values(X, n_values=12):
    """
    Make cumulative-value path segments.

    For block n with start t = 12n, define the 12-value sequence

        x_t, x_{t+1}, ..., x_{t+11}.

    The corresponding cumulative path is

        0,
        x_t,
        x_t + x_{t+1},
        ...,
        sum_{k=0}^{11} x_{t+k}.

    Therefore each segment has 13 nodes and 12 increments.
    """
    X = np.asarray(X, dtype=np.float64)

    step = n_values
    n_nodes = n_values + 1
    segments = []

    for start in range(0, len(X) - n_values + 1, step):
        block = X[start:start + n_values]

        seg = np.zeros((n_nodes, X.shape[1]), dtype=X.dtype)
        seg[1:, :] = np.cumsum(block, axis=0)

        segments.append(seg)

    return np.asarray(segments)


def sigma_sig_segments(seg_raw: np.ndarray, include_initial: bool = True) -> float:
    """
    Signature scale:
    collect all nodes from all segments and compute

        sigma^2 = E ||U - V||^2
                = 2 ( E||U||^2 - ||EU||^2 )

    where U and V are sampled uniformly from all cumulative-path nodes.
    """
    seg_raw = np.asarray(seg_raw, dtype=np.float64)

    if include_initial:
        Z = seg_raw.reshape(-1, seg_raw.shape[-1])
    else:
        Z = seg_raw[:, 1:, :].reshape(-1, seg_raw.shape[-1])

    mean_norm2 = np.mean(np.sum(Z * Z, axis=1))
    mu = np.mean(Z, axis=0)
    mu_norm2 = float(mu @ mu)

    sigma2 = 2.0 * (mean_norm2 - mu_norm2)
    return float(np.sqrt(max(sigma2, 0.0)))


# ============================================================
# Parameters
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate slow-variable data from the two-scale Lorenz-96 model."
    )
    parser.add_argument(
        "seed",
        type=int,
        help="Random seed for the initial perturbation.",
    )
    parser.add_argument(
        "--F",
        type=float,
        default=10.0,
        help="Constant forcing parameter F. Default: 10.0.",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=1.0,
        help="Two-scale coupling parameter h. Default: 1.0.",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/l96.npz",
        help="Output npz file. Default: data/l96.npz.",
    )
    return parser.parse_args()


args = parse_args()

K = 10          # number of slow variables
L_SEG = 12

J = 8
F = args.F
h = args.h
c = 10.0
b = 10.0
DT = 0.01
#J = 8           # number of fast variables per slow variable
#F = 8.0
#h = 1.0
#c = 10.0
#b = 10.0
#DT = 0.005
N_STEPS = 100000
SPINUP = 25000
SEED = args.seed

DATA_FILE = args.data_file

os.makedirs(os.path.dirname(DATA_FILE) or ".", exist_ok=True)
os.makedirs("figs", exist_ok=True)


# ============================================================
# Simulation
# ============================================================

print("Two-scale Lorenz-96")
print("K:", K)
print("J:", J)
print("F, h, c, b:", F, h, c, b)
print("DT:", DT)
print("N_STEPS:", N_STEPS)
print("SPINUP:", SPINUP)

X = simulate_l96_2scale(
    K=K,
    J=J,
    F=F,
    h=h,
    c=c,
    b=b,
    dt=DT,
    n_steps=N_STEPS,
    spinup=SPINUP,
    seed=SEED,
)

print("X shape:", X.shape)
print("X min/mean/max:", X.min(), X.mean(), X.max())


# ============================================================
# Segment construction and normalization
# ============================================================

seg_raw = make_segments_cumulative_values(X, n_values=L_SEG)

SEG_SCALE = sigma_sig_segments(seg_raw, include_initial=True)

if SEG_SCALE <= 0:
    raise ValueError("SEG_SCALE must be positive.")

seg = seg_raw / SEG_SCALE

print("seg_raw shape:", seg_raw.shape)
print("seg shape:", seg.shape)
print("SEG_SCALE:", SEG_SCALE)
print("SEG_SCALE_KIND: pairwise_node_distance")


# ============================================================
# Save
# ============================================================

np.savez(
    DATA_FILE,
    X=X,
    seg=seg,
    K=K,
    J=J,
    L_SEG=L_SEG,
    F=F,
    h=h,
    c=c,
    b=b,
    DT=DT,
    N_STEPS=N_STEPS,
    SPINUP=SPINUP,
    SEED=SEED,
    SEG_SCALE=SEG_SCALE,
    SEG_SCALE_KIND="pairwise_node_distance",
    MODEL="two_scale_lorenz96_slow_only",
)

print("Saved:", DATA_FILE)


# ============================================================
# Diagnostics and plots
# ============================================================

t = np.arange(X.shape[0]) * DT

# ------------------------------------------------------------
# 1. Time series of selected slow variables
# ------------------------------------------------------------
plt.figure(figsize=(9, 4))

for j in range(min(3, X.shape[1])):
    plt.plot(t[:5000], X[:5000, j], label=f"X_{j}")

plt.xlabel("Model time")
plt.ylabel("slow variable")
#plt.title("Two-scale Lorenz-96 slow variables, first 5000 steps")
plt.legend()
plt.tight_layout()
plt.savefig("figs/l96_2scale_timeseries_first5000.pdf", dpi=300, bbox_inches="tight")
#plt.show()


# ------------------------------------------------------------
# 2. Space-time image of slow variables
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.imshow(
    X[:5000].T,
    aspect="auto",
    origin="lower",
    extent=[t[0], t[4999], 0, X.shape[1] - 1],
)

plt.colorbar(label="X_k")
plt.xlabel("Model time")
plt.ylabel("slow variable index k")
#plt.title("Two-scale Lorenz-96 slow-variable space-time diagram")
plt.tight_layout()
plt.savefig("figs/l96_2scale_spacetime_first5000.pdf", dpi=300, bbox_inches="tight")
#plt.show()


# ------------------------------------------------------------
# 3. One example of 13-node normalized cumulative path segment
# ------------------------------------------------------------
seg_id = 0
path = seg[seg_id]
tau = np.arange(path.shape[0])

plt.figure(figsize=(8, 4))

for j in range(min(3, path.shape[1])):
    plt.plot(tau, path[:, j], marker="o", label=f"X_{j}")

plt.xlabel("node index within segment")
plt.ylabel("scaled cumulative value")
#plt.title("Example normalized 13-node slow-variable path segment")
plt.legend()
plt.tight_layout()
plt.savefig("figs/l96_2scale_example_segment.pdf", dpi=300, bbox_inches="tight")
#plt.show()


# ------------------------------------------------------------
# 4. Increment size distribution
# ------------------------------------------------------------
dseg = seg[:, 1:, :] - seg[:, :-1, :]
increment_norms = np.linalg.norm(dseg.reshape(-1, X.shape[1]), axis=1)

plt.figure(figsize=(6, 4))
plt.hist(increment_norms, bins=50)
plt.xlabel("||Δx||")
plt.ylabel("count")
#plt.title("Distribution of normalized increment norms")
plt.tight_layout()
plt.savefig("figs/l96_2scale_increment_norm_hist.pdf", dpi=300, bbox_inches="tight")
#plt.show()

print("increment norm mean:", increment_norms.mean())
print("increment norm median:", np.median(increment_norms))
print("increment norm max:", increment_norms.max())
