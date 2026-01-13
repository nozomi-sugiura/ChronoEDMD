import sys, math
import numpy as np
#import sig_inv as si
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy.optimize import brentq
from tqdm import tqdm
import concurrent.futures
from scipy.optimize import minimize
from functools import partial
from scipy.optimize import minimize_scalar

np.set_printoptions(precision=2)

def compute_flat_area_weights(valid_indices, ny=89, nx=180):
    """
    フラット化された有効格子点に対応する面積重みベクトルを返す。

    Parameters
    ----------
    valid_indices : array_like
        ny*nx にフラット化された全格子点のうち、有効な点のインデックス
    ny : int
        緯度方向の格子点数
    nx : int
        経度方向の格子点数

    Returns
    -------
    weights : np.ndarray
        shape = (len(valid_indices), )
        各有効点に対応する cos(緯度) に基づく面積重み
    """
    latitudes = np.linspace(-88, 88, ny)  # 南→北
    coslat = np.cos(np.deg2rad(latitudes))  # shape = (ny,)

    # full shape の重み行列を生成
    weights_2d = np.repeat(coslat[:, np.newaxis], nx, axis=1)  # shape = (ny, nx)
    weights_flat = weights_2d.flatten()  # shape = (ny*nx,)

    # 有効なインデックスのみに絞る
    return np.sqrt(weights_flat[valid_indices])
def rbf_kernel(x, y, sigma):
    return np.exp(-np.sum((x - y)**2) / (2 * sigma**2))

def poly_kernel(x, y, sigma, c=0, d=1):
    return (np.dot(x, y) / sigma**2 + c)**d

def base_kernel_factory(kernel_type="rbf", sigma=None, c=0, d=1):
    if kernel_type == "rbf":
        return partial(rbf_kernel, sigma=sigma)
    elif kernel_type == "poly":
        return partial(poly_kernel, sigma=sigma, c=c, d=d)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

def compute_signature_kernel(args):
    X1, X2, m, i_pad, base_kernel = args
    R_list, _ = signature_kernel(padding(X1, i_pad), padding(X2, i_pad), m, base_kernel=base_kernel)
    return R_list

def parallel_signature_kernel2(X, m, i_pad, base_kernel):
    ntr = X.shape[2]
    M_list = [np.zeros((ntr, ntr)) for _ in range(m + 1)]  # M^{(j)} for j=0..m

    tasks = [
        (X[:, :, itr1], X[:, :, itr2], m, i_pad, base_kernel)
        for itr1 in range(ntr)
        for itr2 in range(itr1 + 1)
    ]

    print("Computing M^{(j)} matrices for each order j:")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(compute_signature_kernel, tasks), total=len(tasks)))

    idx = 0
    for itr1 in range(ntr):
        for itr2 in range(itr1 + 1):
            R_list = results[idx]
            for j in range(m + 1):
                val = R_list[j] 
                M_list[j][itr1, itr2] = val
                if itr1 != itr2:
                    M_list[j][itr2, itr1] = val
            idx += 1
    #check            
    M = sum(M_list)
    is_pd = is_positive_definite(M)
    print("#Is the Gram matrix positive definite?:", is_pd)
    #save
    np.savez("Mw_orders.npz", **{f"M_order{j}": M_list[j] for j in range(m + 1)})
    print("All Mw^{(j)} saved.")
    return #M_list

def suffix_sum_strict(A):
    # S[i,j] = sum_{p>=i, q>=j} A[p,q]  (inclusive suffix sum)
    S = np.cumsum(np.cumsum(A[::-1, ::-1], axis=0), axis=1)[::-1, ::-1]
    # strict: sum_{p>i, q>j} A[p,q] = S[i+1, j+1]
    out = np.zeros_like(A)
    out[:-1, :-1] = S[1:, 1:]
    return out

def signature_kernel(X, Y, m, base_kernel):
    # (長さが異なる場合も想定するのが本来は自然です)
    nx = X.shape[0]
    ny = Y.shape[0]

    kij = np.zeros((nx, ny), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            kij[i, j] = base_kernel(X[i], Y[j])

    K = kij[1:, 1:] + kij[:-1, :-1] - kij[:-1, 1:] - kij[1:, :-1]

    A = K.copy()                      # A^1
    cum_prev = A.sum()
    level_list = [1.0, cum_prev]      # 0次=1, 1次=ΣA^1

    for d in range(2, m + 1):
        Q = suffix_sum_strict(A)      # Q[i,j] = Σ_{i'>i,j'>j} A[i',j']
        A = K * (1.0 + Q)             # A^d
        cum = A.sum()                 # ΣA^d (≤d の累積)
        level_list.append(cum - cum_prev)  # ちょうど d 次
        cum_prev = cum

    total = 1.0 + cum_prev            # = 1 + ΣA^m
    return level_list, total
def signature_kernel_NG(X, Y, m, base_kernel):
    """
    Compute signature kernel up to order m and return the kernel matrix of each order.
    
    Returns:
        R_list: list of floats, each R_list[l-1] is the contribution from order l.
        Total: scalar, sum over all orders.
    """
    n, d = X.shape
    kij = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            kij[i, j] = base_kernel(X[i], Y[j]) 
    K = kij[1:, 1:] + kij[:-1, :-1] - kij[:-1, 1:] - kij[1:, :-1]
    A = K.copy()
    R_list = [np.sum(A)]  # order 1
    for l in range(2, m + 1):
        Q = np.cumsum(np.cumsum(A, axis=0), axis=1)
        Qshift = shift_mat(Q, 1, 1)
        A = K * (1 + Qshift)
        R_list.append(np.sum(A))
    R_list = [1.0] + R_list  # 0次成分は常に1
    return R_list, sum(R_list)
def signature_kernel_lm(X, Y, m, lmx, lmy, base_kernel):
    diag = False
    n, d = X.shape
    n_, d_ = Y.shape
    lx, ly = n - 1, n_ - 1

    kij = np.zeros((n, n_), dtype=np.float64)
    for i in range(n):
        for j in range(n_):
            kij[i, j] = base_kernel(X[i], Y[j]) * lmx * lmy

    K = kij[1:, 1:] + kij[:-1, :-1] - kij[:-1, 1:] - kij[1:, :-1]
    A = K.copy()

    if diag:
        r_old = 0
        r = 1 + np.sum(A)
        print("#Kernel", 1, r - r_old)
        r_old = r

    for l in range(2, m + 1):
        Q = np.cumsum(np.cumsum(A, axis=0), axis=1)
        Qshift = shift_mat(Q, 1, 1)
        A = K * (1 + Qshift)
        if diag:
            r = 1 + np.sum(A)
            print("#Kernel", l, r - r_old)
            r_old = r

    R = 1 + np.sum(A)
    return R

def padding(X, p):
    n, d = X.shape
    lx = n - 1
    X2 = np.zeros(((p + 1) * lx + 1, d))
    X2[0, :] = X[0, :]
    for i in range(lx):
        for j in range(p + 1):
            X2[(p + 1) * i + 1 + j, :] = ((j + 1) * X[i + 1, :] + (p - j) * X[i, :]) / (p + 1)
    return X2

def shift_mat(Q, s1, s2):
    n, n_ = Q.shape[0] + s1, Q.shape[1] + s2
    Qd = np.zeros((n, n_), dtype=np.float64)
    Qd[s1:, s2:] = Q
    return Qd[:-s1, :-s2]

def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False
def sigma_sig_numpy(X: np.ndarray, I: int = 12) -> float:
    """
    Sig 用: 全ての (i1,i2,j1,j2) を使う
      U = X[i, :, j] を (i,j) 一様にサンプルしたとき
      sigma^2 = E||U - V||^2 = 2(E||U||^2 - ||EU||^2)
    X: (n, d, ntr)
    """
    X = np.asarray(X, dtype=np.float64)
    I = int(I)
    X = X[:I, :, :]                 # (I, d, ntr)
    I0, d, ntr = X.shape

    Z = X.transpose(0, 2, 1).reshape(I0 * ntr, d)   # (I*ntr, d)

    mean_norm2 = np.mean(np.sum(Z * Z, axis=1))     # scalar
    mu = np.mean(Z, axis=0)                         # (d,)
    mu_norm2 = float(mu @ mu)

    sigma2 = 2.0 * (mean_norm2 - mu_norm2)
    return float(np.sqrt(max(sigma2, 0.0)))

#def calc_sig_data(X0):
#    """
#    X0: numpy array of shape (n, d, ntr)
#        n: number of time steps (e.g., 13)
#        d: spatial dimension
#        ntr: number of paths
#    Returns:
#        sigma: average within-path pairwise distance
#    """
#    n, d, ntr = X0.shape
#    sig_data = 0.0
#    num_data = 0
#    for j in range(ntr):
#        for i1 in range(n):
#            for i2 in range(i1 + 1, n):
#                diff = X0[i1, :, j] - X0[i2, :, j]
#                sig_data += 2 * np.sum(diff ** 2)
#                num_data += 2
#    sigma = np.sqrt(sig_data / num_data)
#    return sigma
if __name__ == "__main__":
    m = 7
    i_pad = 1

    X0 = np.load('sst_path.npy')[:, :, :]
    #X0.shape; 13 10988 170
    valid_indices = np.load('valid_indices.npy')
    wt = compute_flat_area_weights(valid_indices, ny=89, nx=180)
    X0 = X0 * wt[np.newaxis, :, np.newaxis]  # broadcasting
#    X0 = np.cumsum(X0, axis=0)/np.sqrt(13.) #Cumulated SST (B)
    

    # 1年=May..Apr の12か月だけにする（最後の1点=翌Mayを捨てる）
    X12 = X0[:12, :, :]                      # shape (12, d, ntr)
    # 先頭に 0 を挿入して累積（13点パス）
    Z0 = np.zeros((1, X12.shape[1], X12.shape[2]), dtype=X12.dtype)  # (1,d,ntr)
    Z  = np.concatenate([Z0, np.cumsum(X12, axis=0)], axis=0)        # (13,d,ntr)
#    sigma_data = calc_sig_data(Z)
    sigma_data = sigma_sig_numpy(Z,I=12)
    print("#sigma_data", sigma_data)
    np.save("sigma",sigma_data)

    n, d, ntr = Z.shape
    print("#Z.shape;", n, d, ntr)

    kernel_type = "rbf"
    base_kernel_with_sigma = base_kernel_factory(kernel_type=kernel_type, sigma=sigma_data, c=0, d=1)
    
    parallel_signature_kernel2(Z, m, i_pad, base_kernel_with_sigma)


