import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cmocean
import warnings, sys
from matplotlib.animation import FuncAnimation, PillowWriter
warnings.filterwarnings("ignore", category=RuntimeWarning)
def efld_prd(eigval):
    log_eig = np.log(eigval)
    efold = 1 / log_eig.real if log_eig.real != 0.0 else np.inf
    period = np.abs(2 * np.pi / log_eig.imag) if log_eig.imag != 0.0 else np.inf
    return efold, period

# --- ファイル読み込み ---
mode = int(sys.argv[1])
print(mode)
ny, nx = 89, 180
data = np.load(f"frames/koopman_modes_{mode:03d}.npz")
Xi = np.ma.masked_array(data["Xi_data"], mask=data["Xi_mask"])
eigv = data["eigv"]
efold, period =  efld_prd(eigv)
print(Xi.shape)
Xi = Xi.reshape(ny, nx)
vmax = 0.9*max(np.abs(Xi.real).max(), np.abs(Xi.imag).max())
vmin = -vmax
#Energy = np.linalg.norm(Xi.filled(0))**2  # マスク部分を 0 とみなす

latitudes = np.linspace(-88, 88, ny)
coslat = np.cos(np.deg2rad(latitudes))  # shape = (ny,)
weights_2d = np.repeat(coslat[:, np.newaxis], nx, axis=1)  # shape = (ny, nx)
valid_mask = ~Xi.mask  # shape = (ny, nx)
Energy = (np.abs(Xi)**2 * weights_2d)[valid_mask].sum()/weights_2d[valid_mask].sum()

# --- 描画準備 ---
# メイン図
fig, ax = plt.subplots(figsize=(8, 4))
cmap = cmocean.cm.balance.copy()
#cmap = cmocean.cm.curl.copy()
cmap.set_bad(color="#654321")
xticks = np.linspace(0, nx-1, 7, dtype=int)
yticks = np.linspace(0, ny-1, 5, dtype=int)
xticklabels = np.linspace(0, 360, 7).astype(int)
yticklabels = np.linspace(-90, 90, 5).astype(int)

im = ax.imshow(Xi.real, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
ax.set_xticks(xticks, labels=xticklabels, fontsize=9)
ax.set_yticks(yticks, labels=yticklabels, fontsize=9)

cbar = plt.colorbar(im)
cbar.set_label("Amplitude")
title_text = ax.set_title(
    f"Koopman Mode {mode} Energy {Energy:.2e}$\mathrm{{K}}^2$\n efld {efold:.2f} per. {period:.2f}"
)

# 時計盤（位相表示用）
inset_ax = fig.add_axes([0.62, 0.863, 0.09, 0.09], polar=True)
#inset_ax.set_theta_direction(1)  # 反時計回りに
#inset_ax.set_theta_offset(0)  # 0ラジアンを右（3時）に
inset_ax.set_theta_direction(-1)  # 時計回りに
inset_ax.set_theta_offset(np.pi / 2)  # 0ラジアンを上（12時）に
inset_ax.set_xticks([])
inset_ax.set_yticks([])
inset_ax.set_ylim(0, 1)
inset_ax.spines['polar'].set_visible(True)
inset_ax.spines['polar'].set_linewidth(0.3)    
needle, = inset_ax.plot([0, 0], [0, 1], color="gray", lw=1.5)
# --- アップデート関数 ---
# アニメーション更新関数の中に追加
def update(frame):
    phase = np.exp(1j * 2 * np.pi * frame / 60)
    rotated = (Xi * phase).real
    im.set_array(rotated)

    angle = 2 * np.pi * frame / 60  # ラジアン（時計盤表示用）
    needle.set_data([angle, angle], [0, 1])  # 中心→外周の針

    title_text.set_text(
        f"Koopman Mode {mode} Energy {Energy:.2e}$\mathrm{{K}}^2$\n"
        f"efld {efold:.2f} per. {period:.2f}"#  phase={angle/(2*np.pi):.2f} ×2π"
    )
    return [im, needle, title_text]
# --- アニメーション ---
ani = animation.FuncAnimation(fig, update, frames=60, interval=100, blit=False)

plt.tight_layout()
plt.show()
ani.save(
    f"frames/koopman_mode_{mode:03d}.gif",
    writer=PillowWriter(fps=10),
    dpi=100
)
#ani.save(
#    f"frames/koopman_mode_{mode:03d}.mp4",
#    writer="ffmpeg",
#    fps=10,
#    dpi=200,
#    bitrate=10000,  # 高ビットレート
#    extra_args=["-pix_fmt", "yuv420p", "-g", "1", "-movflags", "+faststart"]
#)
#ani.save(f"frames/koopman_mode_{mode:03d}.mp4", fps=10, dpi=200)
#ani.save(f"frames/koopman_mode_{mode:03d}.mp4", fps=6, dpi=400)
plt.close()


