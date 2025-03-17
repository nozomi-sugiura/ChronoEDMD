import sys, math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import brentq
from tqdm import tqdm
import concurrent.futures
from scipy.optimize import minimize
np.set_printoptions(precision=2)
def compute_signature_kernel(args):
    """Compute the signature kernel between two sequences using signature_kernel_optimized."""
    X1, X2, m, i_pad, l1, l2 = args  # タプルを展開
    return signature_kernel_lm(padding(X1, i_pad), padding(X2, i_pad), m, l1, l2)

def parallel_signature_kernel(X, Y, m, i_pad, lmx, lmy):
    ntr = X.shape[2]
    G = np.zeros((ntr, ntr))
    A = np.zeros((ntr, ntr))
    L = np.zeros((ntr, ntr))
    
    tasks_G = [(X[:, :, itr1], X[:, :, itr2], m, i_pad, lmx[itr1], lmx[itr2]) for itr1 in range(ntr) for itr2 in range(itr1 + 1)]
    tasks_A = [(Y[:, :, itr1], X[:, :, itr2], m, i_pad, lmy[itr1], lmx[itr2]) for itr1 in range(ntr) for itr2 in range(ntr)]
    tasks_L = [(Y[:, :, itr1], Y[:, :, itr2], m, i_pad, lmy[itr1], lmy[itr2]) for itr1 in range(ntr) for itr2 in range(itr1 + 1)]

    print("Computing G matrix:")
    with concurrent.futures.ProcessPoolExecutor() as executor_G:
        results_G = list(tqdm(executor_G.map(compute_signature_kernel, tasks_G), total=len(tasks_G)))
        executor_G.shutdown(wait=True)  # 🔥 Gの計算後にプロセス解放

    # Populate G
    idx = 0
    for itr1 in range(ntr):
        for itr2 in range(itr1 + 1):
            G[itr1, itr2] = results_G[idx]
            if itr1 != itr2:
                G[itr2, itr1] = results_G[idx]
            idx += 1

    print("Computing A matrix:")
    with concurrent.futures.ProcessPoolExecutor() as executor_A:
        results_A = list(tqdm(executor_A.map(compute_signature_kernel, tasks_A), total=len(tasks_A)))
        executor_A.shutdown(wait=True)  # 🔥 Aの計算後にプロセス解放

    # Populate A
    idx = 0
    for itr1 in range(ntr):
        for itr2 in range(ntr):
            A[itr1, itr2] = results_A[idx]
            idx += 1

    print("Computing L matrix:")
    with concurrent.futures.ProcessPoolExecutor() as executor_L:
        results_L = list(tqdm(executor_L.map(compute_signature_kernel, tasks_L), total=len(tasks_L)))
        executor_L.shutdown(wait=True)  # 🔥 Lの計算後にプロセス解放

    # Populate L
    idx = 0
    for itr1 in range(ntr):
        for itr2 in range(itr1 + 1):
            L[itr1, itr2] = results_L[idx]
            if itr1 != itr2:
                L[itr2, itr1] = results_L[idx]
            idx += 1
            
    return G, A, L
def parallel_signature_kernel2(X, m, i_pad, lmx):
    ntr = X.shape[2]
    M = np.zeros((ntr, ntr))
    
    tasks_M = [(X[:, :, itr1], X[:, :, itr2], m, i_pad, lmx[itr1], lmx[itr2]) for itr1 in range(ntr) for itr2 in range(itr1 + 1)]

    print("Computing M matrix:")
    with concurrent.futures.ProcessPoolExecutor() as executor_M:
        results_M = list(tqdm(executor_M.map(compute_signature_kernel, tasks_M), total=len(tasks_M)))
        executor_M.shutdown(wait=True)  # 🔥 Mの計算後にプロセス解放

    # Populate G
    idx = 0
    for itr1 in range(ntr):
        for itr2 in range(itr1 + 1):
            M[itr1, itr2] = results_M[idx]
            if itr1 != itr2:
                M[itr2, itr1] = results_M[idx]
            idx += 1

    G = M[:-1,:-1]        
    A = M[1:,:-1]        
    L = M[1:,1:]        
    return G, A, L

def padding(X,p):
#interpolate for midpoints
    n, d = X.shape
    lx = n-1
    X2 = np.zeros(((p+1)*lx+1,d))
    X2[0,:] = X[0,:]
    for i in range(lx):
        for j in range(p+1):
            X2[(p+1)*i+1+j,:] = ((j+1)*X[i+1,:]+(p-j)*X[i,:])/(p+1)
    return X2
def Cum(A):
    """
    Computes cumulative sum along a 1D array.

    Parameters:
    A : ndarray
        A 1D array of length n.

    Returns:
    Q : ndarray
        The cumulative sum array.
    """
#    n = len(A)
#    Q = np.zeros(n, dtype=np.float64)
#    Q[0] = A[0]  # First element remains the same
#    for i in range(1, n):  # Forward loop (not reverse order)
#        Q[i] = Q[i - 1] + A[i]
#    return Q
    return np.cumsum(A)

def Cum2(A):
    """
    Computes cumulative sum along both rows and columns of a 2D array.

    Parameters:
    A : ndarray
        A 2D array of shape (n1, n2).

    Returns:
    Q : ndarray
        The cumulative sum matrix.
    """
#    n1,n2 = A.shape
#    Q = A.copy()
#    for i in range(n2):
#        Q[:,i] = Cum(Q[:,i])
#    for i in range(n1):
#        Q[i,:] = Cum(Q[i,:])
#    return Q
    return np.cumsum(np.cumsum(A, axis=0), axis=1)
def shift_mat(Q,s1,s2):
    """
    Computes Qd from Q by shifting elements correctly.

    Parameters:
    Q : ndarray
        A 2D array (n-1, n-1) representing the cumulative sum matrix.

    Returns:
    Qd : ndarray
        A 2D array (n, n_) where Q is mapped with a shift.
    """
    n, n_ = Q.shape[0] + s1, Q.shape[1] + s2  # `Q` のサイズから `Qd` のサイズを計算
    Qd = np.zeros((n, n_), dtype=np.float64)  # `Qd` の初期化
    Qd[s1:, s2:] = Q  # `Q[i-s1, j-s2]` を `Qd[i, j]` に対応させる
    return Qd[:-s1,:-s2]

def signature_kernel(X, Y, m):
    """
    Optimized computation of the truncated signature kernel up to degree m using Algorithm 3.

    Parameters:
    X : ndarray
        A time series of shape (n, d).
    Y : ndarray
        Another time series of shape (n, d).
    base_kernel : function
        A function k(x, y) that computes the base kernel value for two vectors x and y.
    m : int
        The truncation degree of the signature kernel.

    Returns:
    float
        The computed signature kernel value.
    """
    n, d = X.shape
    n_, d_ = Y.shape
    lx, ly = n - 1, n_ - 1

    # Step 1: Compute kernel differences K
    kij = np.zeros((n, n_), dtype=np.float64)
    for i in range(n):
        for j in range(n_):
            kij[i, j] = base_kernel(X[i], Y[j])

    K = kij[1:, 1:] + kij[:-1, :-1] - kij[:-1, 1:] - kij[1:, :-1]
    A = K.copy()

    # Step 2: Iterate to compute A for m > 1
    for l in range(m - 1):
        Q = np.cumsum(np.cumsum(A, axis=0), axis=1)  # Efficient Cum2
        Qshift = shift_mat(Q, 1, 1)#[:-1, :-1]  # Correct the shape to match A
        A = K * (1 + Qshift)  # Element-wise update

    # Step 3: Compute the final kernel value
    R = 1 + np.sum(A)
    return R

def signature_kernel_lm(X, Y, m, lmx, lmy):
    """
    Optimized computation of the truncated signature kernel up to degree m using Algorithm 3.

    Parameters:
    X : ndarray
        A time series of shape (n, d).
    Y : ndarray
        Another time series of shape (n, d).
    base_kernel : function
        A function k(x, y) that computes the base kernel value for two vectors x and y.
    m : int
        The truncation degree of the signature kernel.

    Returns:
    float
        The computed signature kernel value.
    """
    diag = False
    n, d = X.shape
    n_, d_ = Y.shape
    lx, ly = n - 1, n_ - 1

    # Step 1: Compute kernel differences K
    kij = np.zeros((n, n_), dtype=np.float64)
    for i in range(n):
        for j in range(n_):
            kij[i, j] = base_kernel(X[i], Y[j])*lmx*lmy

    K = kij[1:, 1:] + kij[:-1, :-1] - kij[:-1, 1:] - kij[1:, :-1]
    A = K.copy()
    if diag==True:
        r_old = 0
        r = 1 + np.sum(A)
        print("#Kernel",1,r-r_old)
        r_old = r

    # Step 2: Iterate to compute A for m > 1
    for l in range(2,m+1):
        Q = np.cumsum(np.cumsum(A, axis=0), axis=1)  # Efficient Cum2
        Qshift = shift_mat(Q, 1, 1)#[:-1, :-1]  # Correct the shape to match A
        A = K * (1 + Qshift)  # Element-wise update
        if diag==True:
            r = 1 + np.sum(A)
            print("#Kernel",l,r-r_old)
            r_old = r

    # Step 3: Compute the final kernel value
    R = 1 + np.sum(A)
    return R

def save_gram_matrix(G, filename="gram_matrix.png", title="Gram Matrix Heatmap", vmin=None, vmax=None, cmap="coolwarm"):
    """
    Saves a heatmap of the Gram matrix to a file without grid margins.

    Parameters:
    G : ndarray
        Gram matrix (n x n).
    filename : str
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


def plot_gram_matrix(G, title="Gram Matrix Heatmap", vmin=None, vmax=None, cmap="coolwarm"):
    """
    Plots a heatmap of the Gram matrix without grid margins.

    Parameters:
    G : ndarray
        Gram matrix (n x n).
    title : str
        Title of the heatmap.
    vmin : float, optional
        Minimum value for color scale (default: auto).
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
    plt.show()

def is_positive_definite(matrix):
    """
    Checks if a given matrix is positive definite.
    A matrix is positive definite if all its eigenvalues are positive.
    
    Parameters:
    matrix (ndarray): The matrix to check.
    
    Returns:
    bool: True if the matrix is positive definite, False otherwise.
    """
    try:
        # Compute the Cholesky decomposition
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def signature_kernel_coro49(X, Y, m, base_kernel): #Corollary_4.9
    """
    Computes the truncated signature kernel up to degree m using nested loop structure with caching.

    Király, Franz J., and Harald Oberhauser.
    "Kernels for sequentially ordered data." Journal of Machine Learning Research 20.31 (2019): 1-45.

    Parameters:
    X : ndarray
        A time series of shape (n, d).
    Y : ndarray
        Another time series of shape (n, d).
    m : int
        The truncation degree of the signature.
    base_kernel : function
        A function k(x, y) that computes the base kernel value for two vectors x and y.

    Returns:
    float
        The signature kernel value between X and Y.
    """
    n, d = X.shape
    n_, d_ = Y.shape

    # Compute and store kernel differences
    kernel_diff_cache = np.zeros((n - 1, n_ - 1), dtype=np.float64)
    for i in range(n - 1):
        for j in range(n_ - 1):
            kernel_diff_cache[i, j] = (base_kernel(X[i + 1], Y[j + 1]) +
                                       base_kernel(X[i], Y[j]) -
                                       base_kernel(X[i], Y[j + 1]) -
                                       base_kernel(X[i + 1], Y[j]))

    # Recursive function for m > 1
    def compute_kernel(m, i_start, j_start):
        if m == 1:
            # Base case: return cached kernel difference
            return kernel_diff_cache[i_start, j_start]
        
        # Accumulate kernel for all subsequent pairs
        result = 1  # "1 +" in the recursive kernel
        for i_next in range(i_start + 1, n - (m - 1)):
            for j_next in range(j_start + 1, n_ - (m - 1)):
                result += compute_kernel(m - 1, i_next, j_next)
        
        return kernel_diff_cache[i_start, j_start] * result

    # Initialize result
    result = 1  # "1 +" in the kernel formula
    for i1 in range(n - 1):
        for j1 in range(n_ - 1):
            result += compute_kernel(m, i1, j1)

    return result
def func_psi(x,a=1,C=1000):
    if x<=C:
        f = x
    else:
        f = C+C**(1+a)*(C**(-a)-x**(-a))/a
    return f
def func_lmda(lmda,X,m):
    k1 = signature_kernel_lm(X,X, m, lmda, lmda)
    return k1
def func(lmda,X,m,p):
    return func_lmda(lmda,X,m)-p
# Example RBF Kernel
#def base_kernel(x, y, c=0, d=1, sigma=10000):
def base_kernel(x, y, c=0, d=1, sigma=1.45e7**0.5):
    """Radial Basis Function (RBF) kernel."""
#    print(np.sum(x*y)/1e7)
    return np.exp(-np.sum((x - y)**2) / (2 * sigma**2))
#    return (np.dot(x,y)/sigma**2+c)**d
def target_function(m,x=1.0):
    """ 指定した次数ごとの寄与関数 f(m) """
    return (x**m)*math.factorial(m)**(-2)
     # 例: 指数関数的減衰

def compute_kernel_contributions(X, m, lmx, lmy):
    """ 各次数ごとのカーネル寄与を計算 """
    contributions = [1]#of the 0-th iterated integral
    r_old = 0
    for l in range(1, m + 1):
        r = signature_kernel_lm(X, X, l, lmx, lmy)  # l次のカーネル計算
        contributions.append(r - r_old)
        r_old = r
    return np.array(contributions)

def loss_function(lmda, X, m, target_func):
    """ スケール係数を最適化するための損失関数 """
    lmx, lmy = lmda, lmda  # スケール係数を適用
    contributions = compute_kernel_contributions(X, m, lmx, lmy)
    target_values = np.array([target_func(l) for l in range(0, m + 1)])
    
    # 対数をとる（値が0や負にならないように注意、通常は微小値を足すなど）
    epsilon = 1e-10  # ゼロ除け（必要に応じて調整）
    log_contributions = np.log(contributions + epsilon)
    log_target_values = np.log(target_values + epsilon)
    
    # ログスケールでの最小二乗誤差
    return np.sum((log_contributions - log_target_values) ** 2)

def optimize_scaling_factor(X, m, target_func):
    """ スケール調整係数 λ を最適化 """
    initial_lmda = 1.0
    res = minimize(loss_function, initial_lmda, args=(X, m, target_func), method='Nelder-Mead')
    return res.x[0]  # 最適なスケール係数 λ を返す
if __name__ == "__main__":
        m = 10
        i_pad = 1
        # Example usage
        X0 = np.load('sst_path.npy')[:,:,:]
        print("#X0 before",X0[0,:,0])
        n, d, ntr  = X0.shape
        print("#X0.shape;",n,d,ntr)
        # スケール調整係数の最適化
        optimal_lambda = optimize_scaling_factor(padding(X0[:,:,-1],i_pad), m, target_function)
        print(f"Optimal Scaling Factor λ: {optimal_lambda:.4e}")
        # 可視化
        kernel_contributions = compute_kernel_contributions(padding(X0[:,:,-1],i_pad), m, optimal_lambda, optimal_lambda)
        target_values = np.array([target_function(l) for l in range(0, m + 1)])
        print(kernel_contributions)
        print(target_values)
        plt.figure(figsize=(8, 5))
        plt.plot(range(0, m+1), kernel_contributions, label="Kernel Contributions", marker='o')
        plt.plot(range(0, m+1), target_values, label="Target Function", linestyle='--')
        plt.xlabel("Kernel Order m")
        plt.ylabel("Contribution")
        plt.yscale('log')  # <= y軸を対数スケールに変更
#        plt.ylim(-0.01, max(max(kernel_contributions), max(target_values)))  # y 軸の最小値を 0 に固定
        #plt.ylim(-0.01, None)#max(max(kernel_contributions), max(target_values)))  # y 軸の最小値を 0 に固定
        plt.title("Optimized Kernel Contributions vs. Target Function")
        plt.legend()
        plt.savefig("scale.png")   # プロットしたグラフをファイルsin.pngに保存する
        plt.close()
        #スケール変換
        X0 = X0*optimal_lambda
        #ロバスト化
        norm2 = np.zeros(ntr)
        for i in range(ntr):
            norm2[i] = func_lmda(1.0,padding(X0[:,:,i],i_pad),m)
        c = np.percentile(norm2, 90) #could be 90 or 95
        print("c",c)
        plt.hist(norm2, bins=50)
        plt.axvline(c, color='r', linestyle='--', label=f'C = {c:.2f} (90th Percentile)')
        plt.xlabel("Squared Signature Norm")
        plt.ylabel("Frequency")
        plt.title("Distribution of Squared Signature Norms with 80th Percentile")
        plt.legend()
        plt.savefig("norm2.png")   # プロットしたグラフをファイルsin.pngに保存する
#        plt.show()
#
#optimal scaling lmda
        lmda_opt = np.zeros(ntr)
        for i in range(ntr):
            k0 = func_lmda(1.0,padding(X0[:,:,i],i_pad),m)
            p0 = func_psi(k0,a=1.0,C=c)
            lmda_opt[i] = brentq(func, 0, 10000, args=(padding(X0[:,:,i],i_pad),m,p0),xtol=2e-15,maxiter=1000)
            print("#lmda_opt",lmda_opt[i],i)
        np.save("lmda",lmda_opt)
        lmda_mean = np.mean(lmda_opt)
        print("#lmda_mean",lmda_mean)
#        for i in range(X0.shape[2]):
#            X0[:,:,i] = X0[:,:,i]*lmda_opt[i]
#            X0[:,:,i] = X0[:,:,i]*lmda_mean #uniform normalization
        
        isft = 1
        X = X0[:,:,:-isft]
        Y = X0[:,:,isft:]
        lmx = lmda_opt[:-isft]
        lmy = lmda_opt[isft:]

        G, A, L = parallel_signature_kernel2(X0, m, i_pad, lmda_opt)    
        is_pd = is_positive_definite(G)
        v1=np.percentile(G, 95)
        v0=np.percentile(G,  5)
        save_gram_matrix(G, filename="G_sst.png", title="G Matrix Heatmap",vmin=v0,vmax=v1)
        np.save("G_sst",G)    
        print("#Is the Gram matrix positive definite?:", is_pd)
        save_gram_matrix(A, filename="A_sst.png", title="A Matrix Heatmap",vmin=v0,vmax=v1)
        np.save("A_sst",A)    
        save_gram_matrix(L, filename="L_sst.png", title="L Matrix Heatmap",vmin=v0,vmax=v1)
        np.save("L_sst",L)    
