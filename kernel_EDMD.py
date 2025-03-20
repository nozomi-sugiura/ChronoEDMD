import sys
from tqdm import tqdm
import numpy as np
import scipy as cp
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
#from sklearn.cluster import SpectralClustering
import matplotlib.cm as cm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
#import concurrent.futures
from multiprocessing import Pool, cpu_count
import os
import matplotlib.ticker as ticker

# ============================================
# 並列計算用のグローバル変数初期化
# ============================================
def initializer_contour(_K, _K_H, _L, _m):
    global K, K_H, L, m
    K = _K
    K_H = _K_H
    L = _L
    m = _m

# ============================================
# 各zに対するlambda_minを計算
# ============================================
def compute_lambda_min_contour(z):
    A = L - z * K_H - np.conj(z) * K + np.abs(z)**2 * np.eye(m)
    eigvals = np.linalg.eigvalsh(A)
    lambda_min = eigvals[0]
    return lambda_min

# ============================================
# 複素平面zを走査してlambda_minを並列計算
# ============================================
def pseudospectra_contour(K_mat, L_mat, grid_size=300):
    m_val = K_mat.shape[0]

    # 実軸・虚軸の範囲を設定
    x_min, x_max = -1.2, 1.2
    y_min, y_max = -1.2, 1.2

    # メッシュグリッドを作成
    x_values = np.linspace(x_min, x_max, grid_size)
    y_values = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x_values, y_values)

    # 複素数z平面の全点を1次元化して並列処理
    zs = xx.flatten() + 1j * yy.flatten()

    # 並列計算
    with Pool(processes=cpu_count(), initializer=initializer_contour,
              initargs=(K_mat, K_mat.T.conj(), L_mat, m_val)) as pool:
        lambda_min_results = list(tqdm(pool.imap(compute_lambda_min_contour, zs), total=len(zs)))

    # 2次元グリッドに戻す
    lambda_min_array = np.array(lambda_min_results).reshape((grid_size, grid_size))

    return xx, yy, lambda_min_array
def plot_pseudospectra_contour(xx, yy, lambda_min_array0, filename="pseudospectra_contour.pdf"):
    lambda_min_array = lambda_min_array0**0.5
    plt.figure(figsize=(8, 6))

    # lambda_min のクリップ処理
    lambda_min_clipped = np.clip(lambda_min_array, 1e-16, None)

    # log10スケールへ変換
    log_lambda_min = np.log10(lambda_min_clipped)

    # levels 作成
    min_level = np.floor(np.min(log_lambda_min))
    max_level = np.ceil(np.max(log_lambda_min))
    levels = np.arange(min_level, max_level + 0.25, 0.25)

    # ===========================
    # 塗りつぶしコンター図
    # ===========================
    contour_filled = plt.contourf(xx, yy, log_lambda_min, levels=levels, cmap='nipy_spectral')

    # ===========================
    # 輪郭線のコンターを追加（オプション）
    # ===========================
    contour_lines = plt.contour(xx, yy, log_lambda_min, levels=levels, colors='black', linewidths=0.5)

    # ===========================
    # カラーバー
    # ===========================
    cbar = plt.colorbar(contour_filled)
    major_ticks = np.arange(min_level, max_level + 1, 1)
    tick_labels = [r"$10^{{{}}}$".format(int(l)) for l in major_ticks]

    cbar.set_ticks(major_ticks)
    cbar.set_ticklabels(tick_labels)
    cbar.set_label(r"$\tau$", fontsize=12)

    # ===========================
    # 単位円の描画
    # ===========================
    theta = np.linspace(0, 2 * np.pi, 500)
    unit_circle_x = np.cos(theta)
    unit_circle_y = np.sin(theta)
    plt.plot(unit_circle_x, unit_circle_y, 'k--', label=None)

    # ===========================
    # 軸の範囲とアスペクト比を固定
    # ===========================
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title(r'Pseudospectra Contour')

    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def sort_eigenvalues(L,Tr,Tl,dt,n2):
    # 固有値の対数を計算
    log_L = np.log(L)
    # 周期の計算（純実数なら np.inf にする）
    periods = np.full_like(L, np.inf, dtype=float)  # デフォルトで無限大
    non_real_idx = log_L.imag != 0  # 虚部が0でないものを選択
    periods[non_real_idx] = 2 * np.pi / np.abs(log_L[non_real_idx].imag)  # 周期計算
    # インデックスの並べ替え
    # 1. 純実数 (periods = inf) のインデックスを取得し、L.real の降順でソート
    real_indices = np.where(periods == np.inf)[0]  # 純実数のインデックス
    real_indices_sorted = real_indices[np.argsort(-L[real_indices].real)]  # L.real が大きい順
    # 2. 周期が有限のものを取得し、周期の絶対値が大きい順にソート
    finite_indices = np.where(periods != np.inf)[0]  # 周期が有限のもの
    finite_indices_sorted = finite_indices[np.argsort(-np.abs(periods[finite_indices]))]  # 期間の大きい順
    # 3. ソート結果を統合
    sort_indices = np.concatenate([real_indices_sorted, finite_indices_sorted])
    # 並べ替え
    L_sorted = L[sort_indices]
    Tr_sorted = Tl[:, sort_indices]
    Tl_sorted = Tr[:, sort_indices]
#    L_sorted[n2:] = 0.0
#    Tr_sorted[:,n2:] = 0.0
#    Tl_sorted[:,n2:] = 0.0
    return L_sorted, Tr_sorted, Tl_sorted

def plot_real_phases(X, nx, ny, vmin, vmax,
                     xticks, yticks, xticklabels, yticklabels,
                     base_title, filename, L, jt, dt):
    """
    位相を変化させた実部を6枚（縦3段×横2列）でプロットする関数
    vmin/vmaxは引数で指定するバージョン

    Parameters:
        X (numpy.ndarray): 固有関数（2Dの複素数データ）
        nx (int): 経度のデータサイズ
        ny (int): 緯度のデータサイズ
        vmin, vmax (float): カラーマップの範囲（ユーザー指定）
        xticks, yticks (list): 軸の目盛りの位置
        xticklabels, yticklabels (list): 軸の目盛りラベル
        base_title (str): タイトルのベース
        filename (str): 保存ファイル名
        L (numpy.ndarray): 固有値配列
        jt (int): 固有値のインデックス
        dt (float): 時間刻み
    """

    # **周期を計算**
    real_log_L = (np.log(L[jt]) / dt).real
    imag_log_L = (np.log(L[jt]) / dt).imag
    if imag_log_L != 0:
        efold  = 1 / real_log_L  # 増幅 [yr]
        period = np.abs(2 * np.pi / imag_log_L)  # 周期 [yr]
        period_text = f" (e-folding: {efold:.1e} yr, Period: {period:.2f} yr)"
    else:
        efold  = 1 / real_log_L  # 増幅 [yr]
        period_text = f" (e-folding: {efold:.1e} yr, Period: ∞ yr)"  # 周期が無限大（

    # 位相リスト（ラジアン）
    phases = [0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6]
    phase_labels = ['0', 'π/6', 'π/3', 'π/2', '2π/3', '5π/6']

    # 描画範囲をスケーリングに合わせる（経度0-360、緯度-90-90）
    extent = [0, 360, -90, 90]

    plt.figure(figsize=(10, 7.5))  # 図のサイズ（縦長）

    cmap_real = plt.get_cmap('coolwarm', 512)
    cmap_real.set_bad(color="#654321")

    for i, (phase, label) in enumerate(zip(phases, phase_labels), 1):
        # 位相回転後の実部
        X_rotated = (X * np.exp(1j * phase)).real.reshape([ny, nx])

        ax = plt.subplot(3, 2, i)
        im = ax.imshow(X_rotated, origin='upper', extent=extent, cmap=cmap_real, aspect='auto',
                       vmin=vmin, vmax=vmax, interpolation='nearest')

        plt.colorbar(im, ax=ax)

        title = f"{base_title} [Phase: {label}]{period_text}"
        ax.set_title(title, fontsize=8)

        # 軸の設定
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=9)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_eigenvalue_log_and_original(L, labels, dt, num_clusters=5, filename_log="eigenvalue_log_plot.pdf", filename_original="eigenvalue_original_plot.pdf"):
    """
    クラスタリング結果を反映し、固有値 L の対数の実部と虚部、および元の固有値の実部と虚部をプロット。
    各点にモード番号を注記し、クラスタごとに色分けする。

    Parameters:
        L (numpy.ndarray): 固有値の配列
        labels (numpy.ndarray): クラスタ番号（各モードに対応）
        dt (float): 時間刻み（年単位）
        num_clusters (int): クラスタ数
        filename_log (str): 対数固有値プロットの保存ファイル名
        filename_original (str): 元の固有値プロットの保存ファイル名
    """
    # **クラスタごとの色を設定**
    cluster_colors = sns.color_palette("tab10", n_colors=num_clusters)  # クラスタごとの色

    # **固有値の対数を計算**
    log_L = np.log(L) / dt
    real_parts_log = np.real(log_L)
    imag_parts_log = np.imag(log_L)

    real_parts_original = np.real(L)
    imag_parts_original = np.imag(L)
    # 固定する描画範囲
    x_min, x_max = -0.1, 0.1
    y_min, y_max = -1, 1

    plt.figure(figsize=(8, 6))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    for cluster in range(num_clusters):
        cluster_mask = (labels == cluster)

        if cluster == 0:
            label_text = "Discarded"
        elif cluster == 1:
            label_text = "Approved"
        else:
            label_text = f"Cluster {cluster}"
            
        x_vals = real_parts_log[cluster_mask]
        y_vals = imag_parts_log[cluster_mask]
        idx_vals = np.where(cluster_mask)[0]

        # ✅ 範囲内データのみ抽出
        in_range_mask = (x_vals >= x_min) & (x_vals <= x_max) & \
            (y_vals >= y_min) & (y_vals <= y_max)

        plt.scatter(x_vals[in_range_mask], y_vals[in_range_mask],
                    color=cluster_colors[cluster], alpha=0.7, label=label_text)

        # ✅ モード番号も範囲内のみ
        for x, y, idx in zip(x_vals[in_range_mask], y_vals[in_range_mask], idx_vals[in_range_mask]):
            plt.text(x, y, str(idx), fontsize=10, ha='right', va='bottom')

            plt.axhline(0, color='k', linewidth=0.5, linestyle='--')
            plt.axvline(0, color='k', linewidth=0.5, linestyle='--')

    plt.xlabel("Re(log(λ) / dt) [1/yr]")
    plt.ylabel("Im(log(λ) / dt) [rad/yr]")
    plt.title("Eigenvalue Log Plot", fontsize=10)
    plt.legend(loc='best',framealpha=0.2)
    plt.grid(True)

    plt.savefig(filename_log, dpi=300, bbox_inches='tight')
    plt.close()

    # **元の固有値のプロット**
    plt.figure(figsize=(8, 6))
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)

    for cluster in range(num_clusters):
        cluster_mask = (labels == cluster)
        
        if cluster == 0:
            label_text = "Discarded"
        elif cluster == 1:
            label_text = "Approved"
        else:
            label_text = f"Cluster {cluster}"
#    # クラスタごとにプロット
#    for cluster in range(num_clusters):
#        cluster_mask = (labels == cluster)
        plt.scatter(real_parts_original[cluster_mask], imag_parts_original[cluster_mask], 
                    color=cluster_colors[cluster], alpha=0.7, label=label_text)
    plt.legend(loc='best',framealpha=0.2)
    # **単位円をプロット**
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
    plt.gca().add_patch(circle)

    # **軸の設定**
    plt.axhline(0, color='k', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='k', linewidth=0.5, linestyle='--')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.title("Original Eigenvalue Plot")
    plt.grid(True)
    plt.legend(loc='center')

    # **画像の保存**
    plt.savefig(filename_original, dpi=300, bbox_inches='tight')
#    plt.show()
    
def plot_reconstructed_field(mask,Phi_x, Xi, valid_indices, nx, ny, years, selected_years, filename_prefix):
    """
    `Phi_x @ Xi` による再構成フィールドを特定の年ごとにプロット（緯度・経度のラベル統一、整数表記）

    Parameters:
        Phi_x (numpy.ndarray): 固有関数行列 (n_time, n_modes)
        Xi (numpy.ndarray): モードごとの重み (n_modes, n_space)
        valid_indices (numpy.ndarray): 有効な空間インデックス
        nx (int): 経度方向のグリッド数
        ny (int): 緯度方向のグリッド数
        years (numpy.ndarray): 時系列データの年リスト
        selected_years (list): プロットする特定の年リスト
        filename_prefix (str): 保存ファイルの名前プレフィックス
    """
    # 再構成された SST フィールド
#    dsst_reconstructed =  Phi_x @ np.diag(L) @ Xi #next year predicted
    nm = Phi_x.shape[1]  # モード数
#    mask = np.ones(nm, dtype=bool)
#    mask[[120, 157, 172]] = False  # 除外するインデックスをFalseに
#    mask = np.zeros(nm, dtype=bool)
#    mask[[156, 167, 172]] = True
    dsst_reconstructed =  Phi_x[:,mask==1] @ Xi[mask==1,:]
    
    r2 = np.zeros(nm)
    for im in range(nm):
        if mask[im] == 1:
            r2[im] = np.sum(np.abs(np.outer(Phi_x[:, im], Xi[im, :]))**2)
    r2 /= np.sum(r2)   

    # 経度・緯度の設定（plot_real_imaginary に合わせる）
    lon = np.linspace(0.5, 359.5, nx, endpoint=True)  # 経度 0.5°E ~ 359.5°E
    lat = np.linspace(89.5, -89.5, ny, endpoint=True)  # 緯度 90°N ~ -90°S
    
    xticks = np.linspace(0, nx-1, 7, dtype=int)  # 経度方向の目盛り
    yticks = np.linspace(0, ny-1, 5, dtype=int)  # 緯度方向の目盛り
    xticklabels = np.linspace(0, 360, 7).astype(int)  # 経度ラベル（整数）
    yticklabels = np.linspace(90, -90, 5).astype(int)  # 緯度ラベル（整数）

    for year in selected_years:
        index = np.where(years == year)[0][0]  # 該当年のインデックスを取得
        field = np.full((ny * nx), fill_value=1e20, dtype=np.float64)
        field[valid_indices] = dsst_reconstructed[index, :].real

        # マスク処理
        field_masked = np.ma.masked_where(field == 1e20, field)

        plt.figure(figsize=(6, 3))
        
        # ★カラーマップ（Spectral）
        cmap = plt.get_cmap('Spectral', 512).reversed()
        cmap.set_bad(color="#654321")  # 深緑 (Dark Green)
        
        im = plt.imshow(field_masked.reshape([ny, nx]), origin='upper', cmap=cmap, aspect='auto',
                        vmin=-3, vmax=30, interpolation='none')  # interpolation を 'none' に設定
#                        vmin=-2, vmax=2, interpolation='none')  # interpolation を 'none' に設定

        plt.colorbar(im, label="SST [$^\circ$C]")
        plt.title(f"Reconstructed annual SST ({year})")

        # 軸の設定（整数ラベルを適用）
        plt.xticks(xticks, labels=xticklabels, fontsize=9)
        plt.yticks(yticks, labels=yticklabels, fontsize=9)

        # グリッド線の追加（オプション）
        plt.grid(True, color='black', linestyle='--', linewidth=0.5)

        # 画像保存
        plt.savefig(f"{filename_prefix}_{year}.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    return r2
def plot_mode_timeseries(Phi_x, L, jt, dt, filename="mode_timeseries.pdf"):
    """
    指定された固有関数 Phi_x[:, jt] の実部と虚部の時系列をプロットし、周期をタイトルに表示し、
    1997年（エルニーニョ年）に縦線を引く。

    Parameters:
        Phi_x (numpy.ndarray): 各時点の固有関数値
        L (numpy.ndarray): 固有値（周期計算に使用）
        jt (int): 表示する固有関数のインデックス
        dt (float): 時間刻み
        filename (str): 画像を保存するファイル名
    """
    n = Phi_x.shape[0]
    years = np.arange(n) + 1850  # x軸の年数（1850年スタート）

    # 実部と虚部の取得
    real_part = Phi_x[:, jt].real
    imag_part = Phi_x[:, jt].imag
    # **周期を計算**
    real_log_L = (np.log(L[jt]) / dt).real
    imag_log_L = (np.log(L[jt]) / dt).imag
    if imag_log_L != 0:
        efold  = 1 / real_log_L  # 増幅 [yr]
        period = np.abs(2 * np.pi / imag_log_L)  # 周期 [yr]
        period_text = f" (e-folding: {efold:.1e} yr, Period: {period:.2f} yr)"
    else:
        efold  = 1 / real_log_L  # 増幅 [yr]
        period_text = f" (e-folding: {efold:.1e} yr, Period: ∞ yr)"  # 周期が無限大（

    plt.figure(figsize=(10, 4))

    # 実部のプロット
    plt.plot(years, real_part, label="Real Part", color='red', linestyle='-')

    # 虚部のプロット
    plt.plot(years, imag_part, label="Imaginary Part", color='purple', linestyle='--')

    # **1997年（エルニーニョ年）に縦線を追加**
    plt.axvline(x=1997, color='black', linestyle=':', linewidth=1.5, label="El Niño 1997")

    plt.xlabel("Year")
    plt.ylabel("Eigenfunction Value")
    plt.title(f"Time Series of Mode {jt} ({period_text})",fontsize=8)  # **周期をタイトルに追加**
    plt.legend()
    plt.grid(True)

    # **1850年を左端に固定**
    plt.xlim(1850, years[-1])

    # 画像保存
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
def save_gram_matrix(G, filename="G_sst.pdf", title="Gram Matrix Heatmap", vmin=None, vmax=None):
    n = G.shape[0]
    years = np.arange(n) + 1850  
    """ グラム行列をヒートマップとして保存 """
    fig, ax = plt.subplots(figsize=(7, 7))
    
    cmap = plt.get_cmap('coolwarm', 512)
    cax = ax.matshow(G, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')  # カラーマップを赤~青に変更

    # 軸の目盛りを適切に間引く
    tick_positions = np.linspace(0, n-1, min(10, n), dtype=int)  # 最大10個の目盛りを設定
    tick_labels = years[tick_positions]

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90)  # x軸ラベルを回転
    ax.set_yticklabels(tick_labels)

    # 軸ラベルを設定
    ax.set_xlabel('$i$-th year')
    ax.set_ylabel('$j$-th year')

    # カラーバー追加
    fig.colorbar(cax, label="$G_{ij}$")

    # タイトル設定
    ax.set_title(title)

    # 画像保存
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
def plot_real_imaginary(X, nx, ny, vmin, vmax, xticks, yticks, xticklabels, yticklabels,
                         real_title, imag_title, filename, L, jt, dt):
    """
    Real Part と Imaginary Part のヒートマップをプロットする関数（周期をタイトルに追加）

    Parameters:
        X (numpy.ndarray): 実部・虚部を含む2Dデータ
        nx (int): 経度のデータサイズ
        ny (int): 緯度のデータサイズ
        vmin (float): カラーマップの最小値
        vmax (float): カラーマップの最大値
        xticks (list): 経度の目盛りの位置
        yticks (list): 緯度の目盛りの位置
        xticklabels (list): 経度の目盛りラベル（整数）
        yticklabels (list): 緯度の目盛りラベル（整数）
        real_title (str): 実部のプロットタイトル
        imag_title (str): 虚部のプロットタイトル
        filename (str): 保存するファイル名
        L (numpy.ndarray): 固有値の配列（周期計算に使用）
        jt (int): 表示する固有関数のインデックス
        dt (float): 時間刻み
    """
    # **周期を計算**
    real_log_L = (np.log(L[jt]) / dt).real
    imag_log_L = (np.log(L[jt]) / dt).imag
    if imag_log_L != 0:
        efold  = 1 / real_log_L  # 増幅 [yr]
        period = np.abs(2 * np.pi / imag_log_L)  # 周期 [yr]
        period_text = f" (e-folding: {efold:.1e} yr, Period: {period:.2f} yr)"
    else:
        efold  = 1 / real_log_L  # 増幅 [yr]
        period_text = f" (e-folding: {efold:.1e} yr, Period: ∞ yr)"  # 周期が無限大（非振動）

    # **タイトルを更新**
    real_title += period_text
    imag_title += period_text

    plt.figure(figsize=(12, 3))  # 図のサイズを指定

    # **カラーマップの選択**
    cmap_real = plt.get_cmap('coolwarm', 512)  # 実部
    cmap_imag = plt.get_cmap('PuOr_r', 512)    # 虚部（反転）
    cmap_real.set_bad(color="#654321")  # 陸地を濃い茶色 (Earth brown) に設定
    cmap_imag.set_bad(color="#654321")  # 陸地を濃い茶色 (Earth brown) に設定

    # **Real 部分の表示**
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(X.real.reshape([ny, nx]), origin='upper', cmap=cmap_real, aspect='auto',
                     vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.colorbar(im1, ax=ax1)
#    proj = ccrs.Mollweide()
#    ax1.coastlines(lw = 0.5)
    ax1.set_title(real_title,fontsize=8)

    # **目盛りの設定**
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels, fontsize=9)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels, fontsize=9)
    # **Imaginary 部分の表示**
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(X.imag.reshape([ny, nx]), origin='upper', cmap=cmap_imag, aspect='auto',
                     vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title(imag_title,fontsize=8)

    # **目盛りの設定**
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels, fontsize=9)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(yticklabels, fontsize=9)

    plt.tight_layout()  # レイアウト調整
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_z_on_complex_plane(zs_array, filename="pseudospectra_plot.pdf"):
    plt.figure(figsize=(6, 6))


    # --- 単位円を描画 ---
    theta = np.linspace(0, 2 * np.pi, 500)
    unit_circle_x = np.cos(theta)
    unit_circle_y = np.sin(theta)
    plt.plot(unit_circle_x, unit_circle_y, 'k--', label='Unit Circle')

    # --- 採用された z とその共役をプロット ---
    plt.scatter(zs_array.real, zs_array.imag, color='red', marker='.', s=3, label='Accepted z')
    plt.scatter(zs_array.real, -zs_array.imag, color='red', marker='.', s=3, label=None)

    # --- 軸や装飾 ---
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')

    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title('Pseudospectra')
    plt.legend()
    plt.axis('equal')  # アスペクト比を1:1に
    plt.grid(True)

    # --- PDFファイルとして保存 ---
    plt.savefig(filename, format='pdf')
#    print(f"プロットをPDFとして保存しました: {filename}")

#    # --- 画面にも表示 ---
#    plt.show()
def initializer(_K, _K_H, _L, _m, _eps2):
    global K, K_H, L, m, eps2
    K = _K
    K_H = _K_H
    L = _L
    m = _m
    eps2 = _eps2

def compute_lambda_min(z):
    A = L - z * K_H - np.conj(z) * K + np.abs(z)**2 * np.eye(m)
    eigvals, eigvecs = np.linalg.eigh(A)
    lambda_min = eigvals[0]
    v_min = eigvecs[:, 0]
    if lambda_min < eps2:
        return (z, lambda_min, v_min)
    else:
        return None
def pseudospectra(K_mat, L_mat, eps=1e-2):
    eps2_val = eps**2
    m_val = K_mat.shape[0]
    r_min, r_max, r_step = 1-0.2, 1.2, 0.004
    r_values = np.arange(r_min, r_max + r_step, r_step)
    theta_values = np.linspace(0, np.pi, 8*180)

    zs = [r * np.exp(1j * theta) for r in r_values for theta in theta_values]

    with Pool(processes=cpu_count(), initializer=initializer,
              initargs=(K_mat, K_mat.T.conj(), L_mat, m_val, eps2_val)) as pool:
        results = list(tqdm(pool.imap(compute_lambda_min, zs), total=len(zs)))

    # Noneを除外
    results = [r for r in results if r is not None]
    zs_array = np.array([r[0] for r in results])
    lambda_min_array = np.array([r[1] for r in results])
    v_min_array = np.array([r[2] for r in results])

    return zs_array, lambda_min_array, v_min_array

def residual1(K,K2,L,Tl):
    m = L.shape[0]
    res = np.zeros(m)
    for j in range(m):
        A = K2 - L[j] * K.T.conj() - np.conj(L[j]) * K + np.abs(L[j])**2 * np.eye(m)
        res0 = Tl[:,j].T.conj() @ Tl[:,j]
        res1 = Tl[:,j].T.conj() @ A @ Tl[:,j]
        res[j] = np.sqrt((res1/res0).real)
#        print("residual1",j,res[j],res0.real,res1.real)
    return res    
def residual2(K2,L,Tl):
    res = np.zeros(L.shape[0])
    for j in range(L.shape[0]):
        res0 = Tl[:,j].T.conj() @ Tl[:,j]
        res1 = Tl[:,j].T.conj() @ K2 @ Tl[:,j]
        res[j] = np.sqrt((res1/res0).real - np.abs(L[j])**2)
#        print("residual2",j,res[j],res0.real,res1.real)
    return res    
def loo_edmd_error(G, A, verbose=False):
    """
    Leave-One-Out (LOO) による Kernel EDMD 誤差評価（RMS）関数
    
    Parameters
    ----------
    G : np.ndarray
        Gram matrix（n_samples × n_samples）
    A : np.ndarray
        Cross Gram matrix（n_samples × n_samples）
    verbose : bool
        各サンプルの誤差を出力するかどうか
    
    Returns
    -------
    rms_error : float
        Leave-One-Out誤差のRMS値
    errors_loo : np.ndarray
        各サンプルごとの誤差（オプション）
    """
    
    n_samples = G.shape[0]
    errors_loo = []
    
    for i in range(n_samples):
        # Leave-One-Outサンプルインデックス
        idx_train = np.delete(np.arange(n_samples), i)
        
        # LOO版 G と A を作成
        G_train = G[np.ix_(idx_train, idx_train)]
        A_train = A[np.ix_(idx_train, idx_train)]
        
        # EDMD 計算（K行列）
        S2, U = np.linalg.eigh(G_train)
        S = np.diag(S2**0.5)
        Sinv = np.linalg.pinv(S)
        K = Sinv @ U.T @ A_train @ U @ Sinv.T
        
        # LOOテストサンプルとの相関
        G_test = G[i, idx_train]
        A_test = A[i, idx_train]
        
        # 再構成誤差を計算
        reconstruction = G_test @ U @ Sinv @ K.T.conj()
        true_value = A_test @ U @ Sinv
        
        # ユークリッドノルムで誤差を記録
        err = np.linalg.norm(reconstruction - true_value)
        errors_loo.append(err)
        
        if verbose:
            print(f"#LOO error of sample {i}: {err:.6f}")
    # RMS誤差
    errors_loo = np.array(errors_loo)
    rms_error = np.sqrt(np.mean(errors_loo**2))
    return rms_error, errors_loo        
# メイン部分の修正
if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    # データ読み込み
    G = np.load("G_sst.npy")
    v1 = np.percentile(G, 95)
    v0 = np.percentile(G, 5)
    save_gram_matrix(G, filename="G_sst.pdf", vmin=v0, vmax=v1)
    A = np.load("A_sst.npy")
    R = np.load("L_sst.npy")
    sst_loaded = np.load('sst_compressed.npy')
    valid_indices = np.load('valid_indices.npy')

    # 設定値
    nx, ny = 360, 180
    n = G.shape[0]
    years = np.arange(n) + 1850

    # 5%と95%のパーセンタイル値を取得
    v1 = np.percentile(G, 95)
    v0 = np.percentile(G, 5)

    # 月リスト (kt=0 -> Jan, kt=11 -> Dec)
    months = ["Annual Mean", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # 経度・緯度の目盛り設定
    xticks = np.linspace(0, nx-1, 7, dtype=int)
    yticks = np.linspace(0, ny-1, 5, dtype=int)
    xticklabels = np.linspace(0, 360, 7, dtype=int)
    yticklabels = np.linspace(90, -90, 5, dtype=int)

    rms_error, errors_loo = loo_edmd_error(G, A, verbose=False)
    print(f"#LOO RMS Error: {rms_err:.6f}")
    # kernel EDMD
    S2, U = np.linalg.eigh(G)
    S = np.diag(S2**0.5)
    Sinv = np.linalg.pinv(S)
    K = Sinv @ U.T @ A @ U @ Sinv.T
    #Err = np.linalg.norm(A.T @ U @ Sinv - G @ U @ Sinv @ K.T.conj())
    #print("#Err of K", Err)
    L, T, Tl = cp.linalg.eig(K,left=True)
    dt = 1.0    ###固有値の並べ替え
    n2 = 50
    L, T, Tl = sort_eigenvalues(L,T,Tl,dt,n2)
    K2 = Sinv @ U.T @ R @ U @ Sinv.T
    res = residual2(K2,L,Tl)
    #
    Phi_x = U @ S @ T
    a = np.zeros(n)
    for i in range(n):
        a[i] = np.vdot(Phi_x[:,i],Phi_x[:,i]).real
    a = a**0.5
    #normalize the eigenfunctions
    for i in range(n):
        Phi_x[:,i] = Phi_x[:,i]/a[i]
    np.save("Phi_x",Phi_x)

    Phi_x_inv = np.linalg.pinv(Phi_x)
    gap = 2e-1
    approved = np.where(res < 0.2, 1, 0)
    inds = []
    for i in range(L.shape[0]):
        if approved[i] == 1:
            inds.append(i)
    plot_eigenvalue_log_and_original(L, approved, dt,num_clusters=2)
    # データ処理
    nt = sst_loaded.shape[0] // 12 - 2
    dsst = np.zeros((nt, sst_loaded.shape[1]))
    n_modes = len(inds)
    vmax = np.zeros(nt)#,dtype=int)
    vmin = np.zeros(nt)#,dtype=int)
    scale= np.zeros(nt)

    # 可視化したい年を指定
    selected_years = list(range(1980, 2020))
#    selected_years = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999]  # 例として5つの年を指定

    for kt in range(13):
      if kt==0:
          for it in range(nt):
              dsst[it, :] = np.mean(sst_loaded[12*it:12*(it+1),:],axis=0)
#              dsst[it, :] = sst_loaded[12*(it+1),:]-sst_loaded[12*it,:]#next_Jan - Jan
          Xi = Phi_x_inv @ dsst
      else:
          for it in range(nt):
              dsst[it, :] = sst_loaded[12*it+(kt-1),:]
#              dsst[it, :] = sst_loaded[12*it+(kt-1),:]-sst_loaded[12*it,:]
          Xi = Phi_x_inv @ dsst
      if kt==0:
      # 再構成されたフィールドをプロット（モードごとにループせず、一度だけ実行）
          r2 = plot_reconstructed_field(approved,Phi_x, Xi, valid_indices, nx, ny, years, selected_years, filename_prefix="frames/reconstructed_sst")
          for i in range(L.shape[0]):
              print(i, L[i].real,L[i].imag, (np.log(L[i])/dt).real, (np.log(L[i])/dt).imag,\
                    2*np.pi/(np.log(L[i])/dt+1e-20).imag, r2[i], res[i] )
#          print(r2)
#        kt = 0
#        for it in range(nt):
#            intensity[jt,it] = np.sum(dsst[it,:]*Xi[jt,:])
      for jt in inds:
        if kt==0:
              scale[jt] = np.sqrt(np.mean(np.abs(Xi)**2)) #時系列を正規化
              abs_max_r = np.percentile(np.abs((Xi[jt,:]/scale[jt]).real), 99)  # 95% パーセンタイルで上限を設定
              abs_max_i = np.percentile(np.abs((Xi[jt,:]/scale[jt]).imag), 99)  # 95% パーセンタイルで上限を設定
              abs_max = max(abs_max_r,abs_max_i)
              vmax[jt] = max(abs_max, 1e-2)  # 最小スケールを 1e-2 に設定し、極端に小さい値を防ぐ
              vmin[jt] = -vmax[jt]  # vmin は vmax の反転
              # **固有関数の時系列プロット（周期を表示）**
              plot_mode_timeseries(Phi_x*scale[jt], L, jt, dt, filename=f"frames/mode_timeseries_{jt:03d}.pdf")
          
        Xi_restored = np.full((nt, ny * nx), fill_value=1e20, dtype=np.complex64)
        Xi_restored[:, valid_indices] = Xi/scale[jt]
        # 可視化用データ
        X = np.ma.masked_where(Xi_restored[jt, :] == 1e20, Xi_restored[jt, :])

        # **現在の kt に対応する月を取得**
        current_month = months[kt]

        # **タイトルのフォーマット**
        title_format = "Real Part onto {:10}"  
        real_title = title_format.format(current_month)
        imag_title = real_title.replace("Real", "Imaginary")

        
        
        # **関数を使ってプロット**
        plot_real_imaginary(X, nx, ny, vmin=vmin[jt], vmax=vmax[jt],
                    xticks=xticks, yticks=yticks,
                    xticklabels=xticklabels, yticklabels=yticklabels,
                    real_title=real_title,imag_title=imag_title,
                    filename=f"frames/sst_mode={jt:03d}-{kt:02d}.pdf",
                    L=L, jt=jt, dt=1.0)
        if kt==0:
            plot_real_phases(2*X, nx, ny,
                             vmin=2*vmin[jt], vmax=2*vmax[jt],
                 xticks=[0, 90, 180, 270, 360],
                 yticks=[-90, -45, 0, 45, 90],
                 xticklabels=[0, 90, 180, 270, 360],
                 yticklabels=[-90, -45, 0, 45, 90],
                 base_title=f"Eigenfunction {jt:03d}",
                 filename=f"frames/eigenmode_phase_maps_{jt:03d}.pdf",
                 L=L, jt=jt, dt=dt)
#    np.savez("intensity",Phi_x.T,inds)

    # 擬スペクトル計算（コンター表示用）
    xx, yy, lambda_min_array = pseudospectra_contour(K, K2, grid_size=600)
    # コンター表示とPDF保存
    plot_pseudospectra_contour(xx, yy, lambda_min_array, filename="pseudospectra_contour.pdf")
