import netCDF4, sys
import numpy as np
import matplotlib.pyplot as plt

# データ読み込み
infile = 'sst.mon.mean.nc'
nc = netCDF4.Dataset(infile, "r")
nt, ny, nx = nc.variables['sst'].shape
print(nx,ny)
# 緯度・経度データを読み取り
lon = nc.variables['lon'][:]  # 経度 (360,)
lat = nc.variables['lat'][:]  # 緯度 (180,)

# 確認
#print("Longitude:", lon.shape, "Sample values:", lon[:])
#print("Latitude:", lat.shape, "Sample values:", lat[:])

# SSTデータを読み込み
sst = np.zeros((nt, ny, nx), dtype=np.float32)
for it in range(nt):
    sst[it, :, :] = nc.variables['sst'][it, :, :]

# マスク作成（時間0のマスクをすべての時間で適用）
mask = (sst[0, :, :] == 1e20)
mask_flat = mask.flatten()  # マスクを1次元化
valid_indices = np.where(~mask_flat)[0]  # マスクされていないインデックスを取得

smon = 6 #start month


# 圧縮データを作成 (nt × 有効ピクセル数)
sst_compressed = sst.reshape(nt, -1)[smon-1:, valid_indices]
np.save('sst_compressed.npy', sst_compressed)
np.save('valid_indices.npy', valid_indices)  # 有効なインデックスも保存

print("Compressed SST shape:", sst_compressed.shape)


lag = 12
dt = lag+1
nxy = sst_compressed.shape[1]#+1
T2 = np.zeros((dt,nxy,nt//lag-1))
for j in range(nt//lag-1):
    print(j,lag*j+(smon-1),lag*j+dt+(smon-1))
    T2[:,:,j] = sst_compressed[lag*j+(smon-1):lag*j+dt+(smon-1),:]
#    #augment time axis
#    T2[:,-1,j] = np.arange(lag*j+(smon-1),lag*j+dt+(smon-1))/12.
np.save('sst_path',T2)

