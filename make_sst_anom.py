import xarray as xr
import numpy as np
import sys

infile = "ersst_v5_all.nc"
ds = xr.open_dataset(infile)
smon = int(sys.argv[1])  # start month (1..12), e.g. 5 means May-start

# -------------------------
# load SST and compress
# -------------------------
sst1 = ds["sst"].squeeze("lev")  # (time, lat, lon) assumed
nt, ny, nx = sst1.shape

sst = sst1.values  # (nt, ny, nx) numpy
sst_2d = sst.reshape(nt, ny * nx)

# mask: drop gridpoints that are NaN at ANY time (keep only fully-valid points)
mask = np.any(np.isnan(sst), axis=0)          # (ny, nx)
mask_flat = mask.reshape(ny * nx)
valid_indices = np.where(~mask_flat)[0]       # 1D indices into ny*nx

# compressed SST in degC
sst_compressed = sst_2d[smon - 1 :, valid_indices]  # shape (T, d)
T, d = sst_compressed.shape

print("Compressed SST shape (raw):", sst_compressed.shape)

# -------------------------
# monthly climatology (past 30 years + current month, NO future)
# -------------------------
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

        # pick indices in [lo, t] with same month m
        # smallest index >= lo that matches month m:
        first = lo + ((m - lo) % 12)
        idx = np.arange(first, t + 1, 12, dtype=int)

        # safety (should not be empty)
        if idx.size == 0:
            clim[t] = X[t]
        else:
            clim[t] = X[idx].mean(axis=0)
    return clim

clim_compressed = monthly_climatology_past30_inclusive(sst_compressed, half_years=30)
anom_compressed = sst_compressed - clim_compressed

# -------------------------
# time array (aligned with sst_compressed)
# -------------------------
time_all = ds["time"].values                 # (nt,)  datetime64[...] のはず
time_cropped = time_all[smon - 1 :]          # sst_compressed と同じクロップ
assert time_cropped.shape[0] == T

# -------------------------
# save outputs
# -------------------------
np.save("valid_indices.npy", valid_indices)

# 保存する "sst_compressed.npy" を anomaly にする（ここが目的）
np.save("sst_compressed.npy", anom_compressed)

# 参考：climatology も保存（必要なら）
np.save("clim_compressed.npy", clim_compressed)

print("Saved anomaly SST shape:", anom_compressed.shape)
print("Saved climatology shape:", clim_compressed.shape)

# -------------------------
# build T2 path tensor from anomaly
# -------------------------
lag = 12
dt = lag + 1
margin = 0

nt_new = anom_compressed.shape[0]
nxy = anom_compressed.shape[1]

T2 = np.zeros((dt, nxy, nt_new // lag - margin), dtype=np.float64)
#for j in range(nt_new // lag - margin):
#    a = lag * j
#    b = lag * j + dt
#    print(j, a, b)
#    T2[:, :, j] = anom_compressed[a:b, :]
#
#np.save("sst_path.npy", T2)
#print("#dim", nt, nx, ny, nxy)
time_path = np.empty((dt, nt_new // lag - margin), dtype=time_cropped.dtype)

for j in range(nt_new // lag - margin):
    a = lag * j
    b = lag * j + dt
#    print(j, a, b)
    T2[:, :, j] = anom_compressed[a:b, :]
    time_path[:, j] = time_cropped[a:b]      # ★同じ a:b を使う

np.save("sst_path.npy", T2)
np.save("sst_time_path.npy", time_path)      # ★追加
print("Saved time_path shape:", time_path.shape)
