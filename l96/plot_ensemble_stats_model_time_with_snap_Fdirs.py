import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Default settings
# ============================================================

DEFAULT_STATS_GLOB = "F*_S*/figs_sig_kiraly_spk_l96/stats_*.npz"
DEFAULT_OUT_DIR = "ensemble_figs"
DEFAULT_SNAP_STATS_GLOB = "F*_S*/figs_snapshot_kedmd_l96/stats_*.npz"

# "mean" or "rms"
# mean: arithmetic ensemble mean
# rms : root-mean-square over seeds
DEFAULT_AGG_MODE = "mean"

# Error bars:
# "std"  : ensemble standard deviation
# "sem"  : standard error of the mean
# "none" : no error bars
DEFAULT_ERROR_MODE = "std"

# Figure title.
# Use None for manuscript-style figures without an in-panel title.
FIG_TITLE = None


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ensemble-averaged L96 prediction skill from stats_*.npz files."
    )
    parser.add_argument(
        "--stats-glob",
        default=DEFAULT_STATS_GLOB,
        help="Glob pattern for stats files. Default: %(default)s",
    )
    parser.add_argument(
        "--snap-stats-glob",
        default=DEFAULT_SNAP_STATS_GLOB,
        help=(
            "Glob pattern for Snapshot EDMD stats files. "
            "Use --no-snap to omit Snapshot EDMD. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--no-snap",
        action="store_true",
        help="Omit Snapshot EDMD even if snap stats files exist.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory. Default: %(default)s",
    )
    parser.add_argument(
        "--agg-mode",
        default=DEFAULT_AGG_MODE,
        choices=["mean", "rms"],
        help="How to aggregate over ensemble members. Default: %(default)s",
    )
    parser.add_argument(
        "--error-mode",
        default=DEFAULT_ERROR_MODE,
        choices=["std", "sem", "none"],
        help="Error band definition. Default: %(default)s",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="RK4 time step used in the L96 simulation. Default: %(default)s",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=12,
        help="Number of RK4 time steps in one prediction block. Default: %(default)s",
    )
    parser.add_argument(
        "--x-axis",
        default="model-time",
        choices=["model-time", "blocks"],
        help=(
            "Horizontal axis. 'model-time' uses lead * block_size * dt; "
            "'blocks' uses lead index. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--skip-sigerr",
        action="store_true",
        help="Do not plot signature-error ensemble figure.",
    )
    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def load_all_stats(pattern, allow_empty=False, label="stats"):
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        if allow_empty:
            print(f"No {label} files found by pattern: {pattern}")
            return []
        raise FileNotFoundError(f"No {label} files found by pattern: {pattern}")

    print(f"Found {label} files:")
    for f in files:
        print(" ", f)

    stats_list = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        stats = {key: data[key] for key in data.files}
        stats["_file"] = f
        stats_list.append(stats)

    return stats_list


def stack_metric(stats_list, key):
    arrs = []
    for stats in stats_list:
        if key not in stats:
            raise KeyError(f"{key} not found in {stats['_file']}")
        arrs.append(np.asarray(stats[key], dtype=float))
    return np.vstack(arrs)


def get_leads_key(stats):
    if "lead" in stats:
        return "lead"
    if "leads" in stats:
        return "leads"
    raise KeyError("Neither 'lead' nor 'leads' found in stats file.")


def check_leads(stats_list):
    key0 = get_leads_key(stats_list[0])
    leads0 = np.asarray(stats_list[0][key0], dtype=int)

    for stats in stats_list[1:]:
        key = get_leads_key(stats)
        leads = np.asarray(stats[key], dtype=int)

        if not np.array_equal(leads, leads0):
            raise ValueError(
                "Lead arrays differ between stats files:\n"
                f"{stats_list[0]['_file']}\n"
                f"{stats['_file']}"
            )

    return leads0


def aggregate(Y, mode="mean"):
    """
    Y shape: (n_seed, n_lead)
    """
    if mode == "mean":
        return np.mean(Y, axis=0)
    if mode == "rms":
        return np.sqrt(np.mean(Y * Y, axis=0))
    raise ValueError("mode must be 'mean' or 'rms'")


def uncertainty(Y, mode="std"):
    """
    Y shape: (n_seed, n_lead)
    """
    if mode is None or mode == "none":
        return None

    if mode == "std":
        return np.std(Y, axis=0, ddof=1) if Y.shape[0] >= 2 else np.zeros(Y.shape[1])
    if mode == "sem":
        if Y.shape[0] >= 2:
            return np.std(Y, axis=0, ddof=1) / np.sqrt(Y.shape[0])
        return np.zeros(Y.shape[1])
    raise ValueError("ERROR_MODE must be 'std', 'sem', or 'none'")


def make_x_axis(leads, dt, block_size, x_axis_mode):
    if x_axis_mode == "model-time":
        x = leads.astype(float) * block_size * dt
        xlabel = "Lead time (model-time units)"
    elif x_axis_mode == "blocks":
        x = leads.astype(float)
        xlabel = f"Lead time ({block_size}-step blocks)"
    else:
        raise ValueError("x_axis_mode must be 'model-time' or 'blocks'")
    return x, xlabel

def plot_metric(x, x_tick_labels, xlabel, curves, ylabel, filename, title=None):
    plt.figure(figsize=(6.6, 4.3))

    # Keep the original Matplotlib color assignment:
    # SIG = blue, SPK = orange, Sig-EDMD = green, Mean = red.
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {}
    for i, item in enumerate(curves):
        color_map[item["label"]] = prop_cycle[i % len(prop_cycle)]

    # ----------------------------
    # 1. Draw uncertainty bands first
    # ----------------------------
    for item in curves:
        label = item["label"]
        mean = item["mean"]
        err = item["err"]

        if err is None:
            continue

        color = color_map[label]

        if label == "SIG":
            alpha = 0.16
            zorder = 1
        elif label == "Sig-EDMD":
            alpha = 0.10
            zorder = 1
        elif label == "Snap-DMD":
            alpha = 0.12
            zorder = 1
        else:
            alpha = 0.14
            zorder = 1

        plt.fill_between(
            x,
            mean - err,
            mean + err,
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=zorder,
        )

    # ----------------------------
    # 2. Draw lines without markers
    # ----------------------------
    for item in curves:
        label = item["label"]
        mean = item["mean"]
        linestyle = item["linestyle"]
        color = color_map[label]

        if label == "SIG":
            linewidth = 2.8
            zorder = 6
        elif label == "Sig-EDMD":
            linewidth = 2.0
            zorder = 7
        elif label == "Snap-DMD":
            linewidth = 2.2
            zorder = 4
        else:
            linewidth = 2.2
            zorder = 5

        plt.plot(
            x,
            mean,
            linestyle=linestyle,
            linewidth=linewidth,
            color=color,
            label=label,
            zorder=zorder,
        )

    # ----------------------------
    # 3. Draw markers separately
    #    Sig-EDMD uses large hollow squares, so it remains visible
    #    even when it coincides with SIG.
    # ----------------------------
    for item in curves:
        label = item["label"]
        mean = item["mean"]
        marker = item["marker"]
        color = color_map[label]

        if label == "SIG":
            plt.plot(
                x,
                mean,
                linestyle="None",
                marker="o",
                markersize=5.5,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=1.2,
                color=color,
                zorder=12,
            )

        elif label == "Sig-EDMD":
            plt.plot(
                x,
                mean,
                linestyle="None",
                marker="s",
                markersize=8.0,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.8,
                color=color,
                zorder=11,
            )

        elif label == "Snap-DMD":
            plt.plot(
                x,
                mean,
                linestyle="None",
                marker=marker,
                markersize=6.0,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=1.4,
                color=color,
                zorder=9,
            )

        else:
            plt.plot(
                x,
                mean,
                linestyle="None",
                marker=marker,
                markersize=5.5,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=1.2,
                color=color,
                zorder=10,
            )

    plt.xticks(x, x_tick_labels)

    if len(x) > 1:
        dx = float(np.min(np.diff(x)))
    else:
        dx = 1.0

    plt.xlim(x[0] - 0.25 * dx, x[-1] + 0.25 * dx)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

    print("Saved:", filename)


def save_ensemble_npz(out_file, leads, x, dt, block_size, x_axis_mode, ensemble_data):
    np.savez(
        out_file,
        lead=leads,
        lead_blocks=leads,
        lead_model_time=leads.astype(float) * block_size * dt,
        x_axis=x,
        dt=dt,
        block_size=block_size,
        x_axis_mode=x_axis_mode,
        **ensemble_data,
    )
    print("Saved:", out_file)


def save_caption(out_file, n_seed, dt, block_size, x_axis_mode):
    one_lead = block_size * dt
    if x_axis_mode == "model-time":
        axis_sentence = (
            f"The horizontal axis is lead time in model-time units; "
            f"one block corresponds to {block_size}Δt = {one_lead:g}."
        )
    else:
        axis_sentence = (
            f"The horizontal axis is lead time in non-overlapping {block_size}-step blocks; "
            f"one block corresponds to {block_size}Δt = {one_lead:g} model-time units."
        )

    caption = (
        f"Two-scale Lorenz--96 slow-variable prediction experiment. "
        f"{axis_sentence} "
        f"Curves show the ensemble mean over {n_seed} independent initial "
        f"perturbations; shaded bands indicate one standard deviation."
    )
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    print("Saved:", out_file)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stats_list = load_all_stats(args.stats_glob, label="path-method stats")
    leads = check_leads(stats_list)

    snap_stats_list = []
    if not args.no_snap:
        snap_stats_list = load_all_stats(
            args.snap_stats_glob,
            allow_empty=True,
            label="Snapshot EDMD stats",
        )
        if len(snap_stats_list) > 0:
            snap_leads = check_leads(snap_stats_list)
            if not np.array_equal(snap_leads, leads):
                raise ValueError(
                    "Lead arrays differ between path-method and Snapshot EDMD stats."
                )

    x, xlabel = make_x_axis(
        leads,
        dt=args.dt,
        block_size=args.block_size,
        x_axis_mode=args.x_axis,
    )

    if args.x_axis == "model-time":
        # Compact labels such as 0.12, 0.24, ..., 1.44.
        x_tick_labels = [f"{v:g}" for v in x]
    else:
        x_tick_labels = [str(v) for v in leads]

    n_seed = len(stats_list)
    n_seed_snap = len(snap_stats_list)
    print("Number of path-method seeds:", n_seed)
    if len(snap_stats_list) > 0:
        print("Number of Snapshot EDMD seeds:", n_seed_snap)
    print("Leads (blocks):", leads)
    print("Lead times (model units):", leads.astype(float) * args.block_size * args.dt)

    methods = [
        ("SIG", "sig", "o", "-"),
        ("SPK", "spk", "^", "-"),
        ("Sig-EDMD", "sigedmd", "s", "--"),
        ("Mean", "mean", "d", ":"),
    ]
    snap_method = ("Snap-DMD", "snap", "x", "-.")

    metric_info = [
        ("rmse", "RMSE", "rmse_ensemble.pdf"),
        ("kpc", "kPC", "kpc_ensemble.pdf"),
    ]
    if not args.skip_sigerr:
        metric_info.append(("error", "signature error", "sigerr_ensemble.pdf"))

    ensemble_save = {}

    for metric_suffix, ylabel, fig_name in metric_info:
        curves = []

        for label, prefix, marker, linestyle in methods:
            key = f"{prefix}_{metric_suffix}"
            Y = stack_metric(stats_list, key)

            y_mean = aggregate(Y, mode=args.agg_mode)
            y_err = uncertainty(Y, mode=args.error_mode)

            ensemble_save[f"{key}_{args.agg_mode}"] = y_mean
            if y_err is not None:
                ensemble_save[f"{key}_{args.error_mode}"] = y_err

            # Also save all seed values for later inspection.
            ensemble_save[f"{key}_all"] = Y

            curves.append(
                {
                    "label": label,
                    "mean": y_mean,
                    "err": y_err,
                    "marker": marker,
                    "linestyle": linestyle,
                }
            )

        if len(snap_stats_list) > 0:
            label, prefix, marker, linestyle = snap_method
            key = f"{prefix}_{metric_suffix}"
            Y = stack_metric(snap_stats_list, key)

            y_mean = aggregate(Y, mode=args.agg_mode)
            y_err = uncertainty(Y, mode=args.error_mode)

            ensemble_save[f"{key}_{args.agg_mode}"] = y_mean
            if y_err is not None:
                ensemble_save[f"{key}_{args.error_mode}"] = y_err
            ensemble_save[f"{key}_all"] = Y

            curves.append(
                {
                    "label": label,
                    "mean": y_mean,
                    "err": y_err,
                    "marker": marker,
                    "linestyle": linestyle,
                }
            )

        plot_metric(
            x,
            x_tick_labels=x_tick_labels,
            xlabel=xlabel,
            curves=curves,
            ylabel=ylabel,
            filename=os.path.join(args.out_dir, fig_name),
            title=FIG_TITLE,
        )

    save_ensemble_npz(
        os.path.join(args.out_dir, "ensemble_stats.npz"),
        leads=leads,
        x=x,
        dt=args.dt,
        block_size=args.block_size,
        x_axis_mode=args.x_axis,
        ensemble_data=ensemble_save,
    )

    save_caption(
        os.path.join(args.out_dir, "figure_caption.txt"),
        n_seed=n_seed,
        dt=args.dt,
        block_size=args.block_size,
        x_axis_mode=args.x_axis,
    )


if __name__ == "__main__":
    main()
