#!/usr/bin/env python3
import glob
import re
import numpy as np
import xarray as xr

PAT = re.compile(r"ersst\.v5\.(\d{6})\.nc$")

def yyyymm_from_path(p: str) -> str:
    m = PAT.search(p)
    if not m:
        raise ValueError(f"Unexpected filename: {p}")
    return m.group(1)

def main():
    files = sorted(glob.glob("ersst.v5.??????.nc"))
    if not files:
        raise FileNotFoundError("No input files: ersst.v5.YYYYMM.nc")

    # 1) decode_times=False で time を数値のまま読み込む（calendar混在を無視）
    dsets = [
        xr.open_dataset(f, decode_times=False)
        for f in files
    ]

    # 2) “time” 次元で連結（ファイル順）
    ds = xr.concat(dsets, dim="time")

    # 3) 月次の time 座標をファイル名から作り直す（例：YYYY-MM-15）
    yyyymm = [yyyymm_from_path(f) for f in files]
    years = np.array([int(s[:4]) for s in yyyymm], dtype=int)
    months = np.array([int(s[4:6]) for s in yyyymm], dtype=int)

    # 月中日=15日で統一（あなたの先行検証と同じ）
    time = np.array([np.datetime64(f"{y:04d}-{m:02d}-15") for y, m in zip(years, months)])
    ds = ds.assign_coords(time=("time", time))

    # 4) 念のため time の重複やソートを保証
    order = np.argsort(ds["time"].values)
    ds = ds.isel(time=order)
    _, idx = np.unique(ds["time"].values, return_index=True)
    ds = ds.isel(time=np.sort(idx))

    ds.to_netcdf("ersst_v5_all.nc")
    print(f"Wrote: ersst_v5_all.nc (nfiles={len(files)}, ntime={ds.sizes['time']})")

if __name__ == "__main__":
    main()


