import xarray as xr
import numpy as np
import sys

def monthly_climatology_past30_inclusive(X: np.ndarray, half_years: int = 30) -> np.ndarray:
    """
    X: (T, d) monthly series in degC (already cropped from start month)
    returns clim: (T, d) where clim[t] is mean over same calendar month
    from max(0,t-30y) .. t inclusive (no future).
    """
    X = np.asarray(X, dtype=np.float64)
    T, d = X.shape
    half = int(half_years) * 12

    clim = np.empty_like(X)
    for t in range(T):
        m = t % 12
        lo = max(0, t - half)

        # smallest index >= lo that matches month m
        first = lo + ((m - lo) % 12)
        idx = np.arange(first, t + 1, 12, dtype=int)

        if idx.size == 0:
            clim[t] = X[t]
        else:
            clim[t] = X[idx].mean(axis=0)
    return clim

def main():
    if len(sys.argv) < 2:
        print("Usage: python make_sst_anom.py <smon:1..12>")
        sys.exit(1)

    infile = "ersst_v5_all.nc"
    smon = int(sys.argv[1])  # start month (1..12)
    if not (1 <= smon <= 12):
        raise ValueError("smon must be in 1..12")

    ds = xr.open_dataset(infile)

    # -------------------------
    # load SST and compress
    # -------------------------
    sst1 = ds["sst"].squeeze("lev")  # (time, lat, lon)
    nt, ny, nx = sst1.shape

    sst = sst1.values  # (nt, ny, nx)
    sst_2d = sst.reshape(nt, ny * nx)

    # mask: drop gridpoints that are NaN at ANY time
    mask = np.any(np.isnan(sst), axis=0)     # (ny, nx)
    valid_indices = np.where(~mask.reshape(ny * nx))[0]

    # crop by start month, then compress
    sst_compressed = sst_2d[smon - 1 :, valid_indices]  # (T, d)
    T, d = sst_compressed.shape
    print("Compressed SST shape (raw):", sst_compressed.shape)

    # -------------------------
    # time array (aligned with sst_compressed)
    # -------------------------
    time_all = ds["time"].values          # (nt,) datetime64
    time_cropped = time_all[smon - 1 :]   # (T,)
    if time_cropped.shape[0] != T:
        raise ValueError(f"time_cropped length {time_cropped.shape[0]} != T={T}")

    # -------------------------
    # monthly climatology (past 30 years + current month, NO future)
    # -------------------------
    clim_compressed = monthly_climatology_past30_inclusive(sst_compressed, half_years=30)
    anom_compressed = sst_compressed - clim_compressed

    # -------------------------
    # save outputs
    # -------------------------
    np.save("valid_indices.npy", valid_indices)
    np.save("sst_compressed.npy", anom_compressed)    # anomaly saved here
    np.save("clim_compressed.npy", clim_compressed)

    print("Saved anomaly SST shape:", anom_compressed.shape)
    print("Saved climatology shape:", clim_compressed.shape)

    # -------------------------
    # build T2 path tensor from anomaly (+ aligned time_path)
    # -------------------------
    lag = 12
    dt = lag + 1
    margin = 0

    nt_new = anom_compressed.shape[0]
    nxy = anom_compressed.shape[1]

    # ★重要: dt 点が必ず取れる窓だけを作る
    ntr = (nt_new - dt) // lag + 1 - margin
    if ntr <= 0:
        raise ValueError(f"Not enough data: nt_new={nt_new}, dt={dt}, lag={lag}, margin={margin}")

    T2 = np.zeros((dt, nxy, ntr), dtype=np.float64)
    time_path = np.empty((dt, ntr), dtype=time_cropped.dtype)

    for j in range(ntr):
        a = lag * j
        b = a + dt  # ★常に b <= nt_new になるよう ntr を定義した

        # デバッグしたいときだけ有効化
        # print("j,a,b =", j, a, b, "slice shape =", anom_compressed[a:b, :].shape)

        T2[:, :, j] = anom_compressed[a:b, :]
        time_path[:, j] = time_cropped[a:b]

    np.save("sst_path.npy", T2)
    np.save("sst_time_path.npy", time_path)

    print("Saved sst_path shape:", T2.shape)          # (13, d, ntr)
    print("Saved time_path shape:", time_path.shape)  # (13, ntr)
    print("#dim", nt, nx, ny, nxy)

if __name__ == "__main__":
    main()
