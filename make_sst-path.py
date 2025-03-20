import netCDF4
import numpy as np
import matplotlib.pyplot as plt

# データ読み込み
infile = 'data/sst.mon.mean.nc'
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

# 圧縮データを作成 (nt × 有効ピクセル数)
sst_compressed = sst.reshape(nt, -1)[:, valid_indices]
np.save('sst_compressed.npy', sst_compressed)
np.save('valid_indices.npy', valid_indices)  # 有効なインデックスも保存

print("Compressed SST shape:", sst_compressed.shape)


lag = 12
dt = lag+1
nxy = sst_compressed.shape[1]#+1
smon = 11 #start month
T2 = np.zeros((dt,nxy,nt//lag-1))
for j in range(nt//lag-1):
    print(lag*j,lag*j+dt)
    T2[:,:,j] = sst_compressed[lag*j+(smon-1):lag*j+dt+(smon-1),:]
#    #augment time axis
#    T2[:,-1,j] = np.arange(lag*j+(smon-1),lag*j+dt+(smon-1))/12.
np.save('sst_path',T2)


# ---- データ復元 ----
sst_loaded = np.load('sst_compressed.npy')
valid_indices = np.load('valid_indices.npy')

# 圧縮データを元の形に復元
sst_restored = np.full((nt, ny * nx), fill_value=1e20, dtype=np.float32)
sst_restored[:, valid_indices] = sst_loaded
sst_restored = sst_restored.reshape(nt, ny, nx)  # 元の形に戻す

# 経度・緯度を仮定して設定（適宜調整してください）
lon = np.linspace(0.5, 359.5, 360, endpoint=True)
lat = np.linspace(89.5, -89.5, 180, endpoint=True)

# 時刻0のデータを抽出してマスク
iy = 2015
sst_time0 = sst_restored[0+12*(iy-1850), :, :]
sst_time1 = sst_restored[12+12*(iy-1850), :, :]
sst_time0_masked = np.ma.masked_where(sst_time0 == 1e20, sst_time1-sst_time0)

# ヒートマップを描画
plt.figure(figsize=(12, 6))
plt.imshow(sst_time0_masked, origin='upper', cmap='rainbow', aspect='auto')
plt.colorbar(label='SST (°C)')
plt.title('Sea Surface Temperature at Time 0 (Masked 1e20)')

# 軸ラベルを設定（ラベルの数を位置と一致させる）
xticks = np.linspace(0, 359, 7, dtype=int)  # 経度インデックスの位置
yticks = np.linspace(0, 179, 5, dtype=int)  # 緯度インデックスの位置
plt.xticks(xticks, labels=np.round(lon[xticks], 1))
plt.yticks(yticks, labels=np.round(lat[yticks], 1))

plt.xlabel('Longitude (°)')
plt.ylabel('Latitude (°)')
plt.show()
