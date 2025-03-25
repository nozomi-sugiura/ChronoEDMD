import math 
import numpy as np
#import sig_inv as si
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import brentq
from tqdm import tqdm
import concurrent.futures
# from scipy.optimize import minimize  # <-- Unused import
from functools import partial
from scipy.optimize import minimize_scalar

np.set_printoptions(precision=2)

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
    X1, X2, m, i_pad, l1, l2, base_kernel = args
    return signature_kernel_lm(padding(X1, i_pad), padding(X2, i_pad), m, l1, l2, base_kernel=base_kernel)

def parallel_signature_kernel2(X, m, i_pad, lmx, base_kernel):
    ntr = X.shape[2]
    M = np.zeros((ntr, ntr))

    tasks_M = [
        (X[:, :, itr1], X[:, :, itr2], m, i_pad, lmx[itr1], lmx[itr2], base_kernel)
        for itr1 in range(ntr)
        for itr2 in range(itr1 + 1)
    ]

    print("Computing M matrix:")
    with concurrent.futures.ProcessPoolExecutor() as executor_M:
        results_M = list(tqdm(executor_M.map(compute_signature_kernel, tasks_M), total=len(tasks_M)))

    idx = 0
    for itr1 in range(ntr):
        for itr2 in range(itr1 + 1):
            M[itr1, itr2] = results_M[idx]
            if itr1 != itr2:
                M[itr2, itr1] = results_M[idx]
            idx += 1

    G = M[:-1, :-1]
    A = M[1:, :-1]
    L = M[1:, 1:]
    return G, A, L

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
def func_psi(x,a=1,C=1000):
    if x<=C:
        f = x
    else:
        f = C+C**(1+a)*(C**(-a)-x**(-a))/a
    return f
def func_lmda(lmda,X,m,base_kernel):
    k1 = signature_kernel_lm(X,X, m, lmda, lmda, base_kernel)
    return k1
def func(lmda,X,m,p,base_kernel):
    return func_lmda(lmda,X,m,base_kernel)-p
def target_function(m,x=1):
    """ 指定した次数ごとの寄与関数 f(m) """
    return (x**m)*math.factorial(m)**(-2)
     # 例: 指数関数的減衰

def compute_kernel_contributions(X, m, lmx, base_kernel):
    """ 各次数ごとのカーネル寄与を計算 """
    contributions = [1]#of the 0-th iterated integral
    r_old = 1
    for l in range(1, m + 1):
        r = signature_kernel_lm(X, X, l, lmx, lmx, base_kernel=base_kernel)  # l次のカーネル計算
        contributions.append(r - r_old)
        r_old = r
        
    return np.array(contributions)
# def compute_all_kernel_contributions(X0, m, lmx, base_kernel):  # <-- Unused function
    """ 各次数ごとのカーネル寄与を計算 """
    all_contributions = []
    for i in range(ntr):
        X_pad = padding(X0[:, :, i], i_pad)
        contributions = compute_kernel_contributions(X_pad, m, lmx, base_kernel)
        all_contributions.append(contributions)
    all_contributions = np.array(all_contributions)  # shape: (ntr, m+1)
    mean_contributions = np.mean(all_contributions, axis=0)
    return mean_contributions

def calc_sig_data(X0):
    n, d, ntr = X0.shape
    num_data = 0
    sig_data = 0.0
    for i in range(n):
        for j in range(ntr):
            sig_data += np.sum(X0[i, :,j] ** 2)
            num_data   += 1
    sig_data /= num_data
    sig_data = sig_data**0.5
    return sig_data
def save_gram_matrix(G, filename="gram_matrix.png", title="Gram Matrix Heatmap", vmin=None, vmax=None, cmap="coolwarm"):
    """
    Saves a heatmap of the Gram matrix to a file without grid margins.

    Parameters:
    G : ndarray
        Gram matrix (n x n). : str
        Name of the output file (supports .png, .pdf, .svg, etc.).
    title : str
        Title of the heatmap.
    vmin : float, optional
        Minimum value for col



    or scale (default: auto).
    vmax : float, optional
        Maximum value for color scale (default: auto).
    cmap : str, optional
        Colormap to use (default: "coolwarm").
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(G, annot=False, fmt=".2f", cmap=cmap, 
                linewidths=0, linecolor="none", cbar=True, vmin=vmin, vmax=vmax, square=True)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    
    # 🔥 ファイルに保存
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Gram matrix heatmap saved as: {filename}")
    
def compute_mean_kernel_contributions(X0, m, base_kernel, i_pad):
    ntr = X0.shape[2]
    contributions_list = np.zeros((ntr, m + 1))
    for i in tqdm(range(ntr)):
        X_pad = padding(X0[:, :, i], i_pad)
        # λ=1で呼び出す
        contributions = compute_kernel_contributions(X_pad, m, lmx=1.0, base_kernel=base_kernel)
#        print(i,contributions)
        contributions_list[i, :] = contributions

    # サンプル間で平均
    mean_contributions = np.mean(contributions_list, axis=0)
    return mean_contributions

def scale_mean_contributions(mean_contributions, lam):
    m_max = len(mean_contributions) - 1
    scale_factors = np.array([lam ** (2 * m) for m in range(m_max + 1)])
    return mean_contributions * scale_factors

def loss_function(lam, mean_contributions, target_func, x1):
    m_max = len(mean_contributions) - 1
    target_values = np.array([target_func(m,x=x1) for m in range(0, m_max + 1)])
    scaled_contributions = scale_mean_contributions(mean_contributions, lam)
#    # 最小二乗誤差
#    return np.sum((scaled_contributions - target_values) ** 2)
# 小さい値を回避するためのイプシロン
    epsilon = 1e-20
    # 対数スケールで比較
    log_scaled = np.log(scaled_contributions + epsilon)
    log_target = np.log(target_values + epsilon)
    # 最小二乗誤差（対数空間での差分）
    return np.sum((log_scaled - log_target) ** 2)    
def scaling_and_normalization(X0, m, i_pad, target_function, base_kernel):
    ntr = X0.shape[2]
    # 1. λ無しで平均寄与を計算
    mean_contributions = compute_mean_kernel_contributions(X0, m, base_kernel_with_sigma, i_pad)
    # 2. λ の最適化
    x1 = 1#0.5
    res = minimize_scalar(
    lambda lam: loss_function(lam, mean_contributions, target_function, x1),
        bounds=(0.01, 100),
        method='bounded'
    )
    optimal_lambda = res.x
    print(f"#Optimal lambda = {optimal_lambda}")
    # 3. スケーリング後の寄与を確認
    scaled_contributions = scale_mean_contributions(mean_contributions, optimal_lambda)
    
    # 4. プロット or ログ出力
    target_values = np.array([target_function(l,x=x1) for l in range(0, m + 1)])
    plt.figure(figsize=(8, 5))
    plt.plot(range(0, m + 1), scaled_contributions, label="Scaled Kernel Contributions", marker='o')
    plt.plot(range(0, m + 1), target_values, label="Target Function", linestyle='--')
    plt.xlabel("Kernel Order m")
    plt.ylabel("Contribution")
    plt.title("Optimized Kernel Contributions vs. Target Function")
    plt.legend()
    plt.yscale('log')
    plt.savefig("scale_lambda_optimized.png")
    plt.close()
    norm2 = np.zeros(ntr)
    for i in range(ntr):
        norm2[i] = func_lmda(optimal_lambda, padding(X0[:, :, i], i_pad), m, base_kernel)

    c = np.percentile(norm2, 90)
    print("#c", c)

    plt.hist(norm2, bins=50)
    plt.axvline(c, color='r', linestyle='--', label=f'C = {c:.2f} (90th Percentile)')
    plt.xlabel("Squared Signature Norm")
    plt.ylabel("Frequency")
    plt.title("Distribution of Squared Signature Norms with 90th Percentile")
    plt.legend()
    plt.savefig("norm2.png")
    plt.close()

    lmda_opt = np.zeros(ntr)
    for i in range(ntr):
        k0 = func_lmda(optimal_lambda, padding(X0[:, :, i], i_pad), m, base_kernel)
        p0 = func_psi(k0, a=1.0, C=c)
        lmda_opt[i] = brentq(func, 0, 10000, args=(padding(X0[:, :, i], i_pad)\
                              , m, p0, base_kernel), xtol=2e-15, maxiter=1000)
    print("#lmda_opt", *lmda_opt)

    np.save("lmda", lmda_opt)
    lmda_mean = np.mean(lmda_opt)
    print("#lmda_mean", lmda_mean)
    
    return optimal_lambda, lmda_opt, lmda_mean, c
if __name__ == "__main__":
    m = 7
    i_pad = 1

    X0 = np.load('sst_path.npy')[:, :, :]
    sigma_data = calc_sig_data(X0)
    print("#sigma_data", sigma_data)

    n, d, ntr = X0.shape
    print("#X0.shape;", n, d, ntr)

    kernel_type = "rbf"
    base_kernel_with_sigma = base_kernel_factory(kernel_type=kernel_type, sigma=sigma_data, c=0, d=1)

    optimal_lambda, lmda_opt, lmda_mean, c\
        = scaling_and_normalization(X0, m, i_pad, target_function, base_kernel_with_sigma)

    G, A, L = parallel_signature_kernel2(X0, m, i_pad, lmda_opt, base_kernel_with_sigma)

    is_pd = is_positive_definite(G)
    print("#Is the Gram matrix positive definite?:", is_pd)
    v1=np.percentile(G, 95)
    v0=np.percentile(G,  5)
    save_gram_matrix(G, filename="G_sst.png", title="G Matrix Heatmap",vmin=v0,vmax=v1)
    np.save("G_sst",G)    
    save_gram_matrix(A, filename="A_sst.png", title="A Matrix Heatmap",vmin=v0,vmax=v1)
    np.save("A_sst",A)    
    save_gram_matrix(L, filename="L_sst.png", title="L Matrix Heatmap",vmin=v0,vmax=v1)
    np.save("L_sst",L)    
