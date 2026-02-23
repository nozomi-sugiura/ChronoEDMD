# ---- build TPI from true SSTA (sst_compressed) ----
import numpy as np
import argparse

def compute_flat_area_weights(valid_indices, ny=89, nx=180, normalize=False):
    latitudes = np.linspace(-88, 88, ny)
    coslat = np.cos(np.deg2rad(latitudes))
    weights_2d = np.repeat(coslat[:, np.newaxis], nx, axis=1)
    weights_flat = weights_2d.reshape(-1)
    w = np.sqrt(weights_flat[valid_indices])
    if normalize:
        w = w / np.sqrt(np.sum(w ** 2))
    return w
parser = argparse.ArgumentParser()
parser.add_argument("start_month", type=int, help="start month of first record (1..12), e.g., 8 for Aug")
args = parser.parse_args()

if not (1 <= args.start_month <= 12):
    raise ValueError("start_month must be in 1..12")
# ---- load ----
sst_compressed = np.load("sst_compressed.npy")      # anomaly, degC
clim_compressed = np.load("clim_compressed.npy")    # climatology, degC
valid_indices = np.load("valid_indices.npy")
wt = compute_flat_area_weights(valid_indices, ny=89, nx=180, normalize=False)

# ---- area-mean RMS correction factor (diagnostic only) ----
d = int(wt.size)

ny, nx = 89, 180
lat = np.arange(-88,  90, 2.0)          # (89,)
lon = np.arange(  0, 360, 2.0)          # (180,) 0–358

# PSL regions in 0–360 lon:
# 145W=215E, 90W=270E, 160W=200E
LAT, LON = np.meshgrid(lat, lon, indexing="ij")  # (ny,nx)

mR1 = (LAT >= 25) & (LAT <= 45) & ( ((LON >= 140) & (LON <= 358)) | (LON <= 215) )
mR2 = (LAT >= -10) & (LAT <= 10) & (LON >= 170) & (LON <= 270)
mR3 = (LAT >= -50) & (LAT <= -15) & ( ((LON >= 150) & (LON <= 358)) | (LON <= 200) )

flatR1 = np.where(mR1.reshape(-1))[0]
flatR2 = np.where(mR2.reshape(-1))[0]
flatR3 = np.where(mR3.reshape(-1))[0]

# map: full-grid flat index -> compressed index (0..d-1), else -1
d = int(wt.size)
pos_of_flat = -np.ones(ny*nx, dtype=np.int64)
pos_of_flat[valid_indices] = np.arange(d, dtype=np.int64)

J1 = pos_of_flat[flatR1]; J1 = J1[J1 >= 0]
J2 = pos_of_flat[flatR2]; J2 = J2[J2 >= 0]
J3 = pos_of_flat[flatR3]; J3 = J3[J3 >= 0]

# area weights for means: cos(lat) = wt^2 (あなたの診断 sum_w2 = sum(wt^2) より)
wt2 = wt**2
A1, A2, A3 = wt2[J1], wt2[J2], wt2[J3]
A1s, A2s, A3s = float(A1.sum()), float(A2.sum()), float(A3.sum())

# sst_compressed: (n_months, d) in degC
sst = sst_compressed  # alias


print("sst shape:", sst.shape)
print("first 13 month norms:", np.linalg.norm(sst[:13], axis=1))
print("min/max first 13:", np.min(sst[:13]), np.max(sst[:13]))
print("allclose first 12 to 0?:", np.allclose(sst[:12], 0.0))
print("allclose month 12 (index 12) to 0?:", np.allclose(sst[12], 0.0))

r1 = (sst[:, J1] * A1).sum(axis=1) / A1s
r2 = (sst[:, J2] * A2).sum(axis=1) / A2s
r3 = (sst[:, J3] * A3).sum(axis=1) / A3s

tpi = r2 - 0.5*(r1 + r3)   # (n_months,)

np.savez("tpi_from_true_sst_ersstv5.npz",
         tpi=tpi, r1=r1, r2=r2, r3=r3,
         J1=J1, J2=J2, J3=J3)
print("saved: tpi_from_true_sst_ersstv5.npz, shape:", tpi.shape)
# ---- write gnuplot-friendly text: time(decimal year), tpi ----
out_txt = "tpi_from_true_sst_ersstv5.txt"

start_year  = 1854
start_month = int(args.start_month)   # 例: 8（argvから読む）
if not (1 <= start_month <= 12):
    raise ValueError("start_month must be in 1..12")

with open(out_txt, "w", encoding="utf-8") as f:
    f.write("# time_decimal_year  tpi_degC\n")
    Y, m = start_year, start_month
    for v in tpi:
        time_decimal = Y + (m - 1) / 12.0
        f.write(f"{time_decimal:10.6f}  {v: .8f}\n")

        # advance 1 month
        m += 1
        if m == 13:
            m = 1
            Y += 1

print("wrote:", out_txt, "start:", f"{start_year}-{start_month:02d}", "N=", len(tpi))
