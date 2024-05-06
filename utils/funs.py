import numpy as np 

def scores(A):
    """ Takes a square matrix A, computes its symmetric and skew symmetric components (time complexity O(d)),
    computes the norm of SYM and SKEWSYM, and finally computes the ratio of the norms.

    Args:
        - A (numpy.ndarray): square numpy matrix.
    Returns:
        - tuple: Symmetric (S) and skew-symmetric (N) scores.
    """

    SYM = .5 * (A + A.T)
    SKEWSYM = .5 * (A - A.T)
     
    S = np.linalg.norm(SYM, 'fro') / np.linalg.norm(A, 'fro')
    N = np.linalg.norm(SKEWSYM, 'fro') / np.linalg.norm(A, 'fro')

    return S, N

def scores_trace(Wq, Wk):
     
    ## (k x n) @ (n x k) -> O(nk^2)
    A = Wq.T @ Wq
    B = Wk.T @ Wk
    C = Wk.T @ Wq

    S = .5 * (1 + (np.einsum('ij,ji->', C, C) / np.einsum('ij,ji->', A, B)))

    return S

def zetas(Wq, Wk):
     
    M = Wq.T @ Wk
    idx = np.einsum('ij,ji->', M, M) 

    return idx

def frobenious_norm(Wq, Wk):
     
    A = Wq.T @ Wq
    B = Wk.T @ Wk
    norm = np.einsum('ij,ji->', A, B)

    return norm