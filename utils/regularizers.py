import numpy as np
import torch

def get_score_norm(A):
    """ Takes a square matrix A, computes its symmetric and skew symmetric components (time complexity O(d)),
    computes the norm of SYM, SKEWSYM and A (np.linalg.norm), and computes the ratio of the norms.

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

def get_score_sum(A):
    """ Takes a square matrix A, computes its symmetric and skew symmetric components (time complexity O(d)),
    computes the square norm of SYM, SKEWSYM and A (sum of the square of matrix entries), 
    and computes the ratio of the norms.

    Args:
        - A (numpy.ndarray): square numpy matrix.
    Returns:
        - tuple: Symmetric (S) and skew-symmetric (N) scores.
    """

    SYM = .5 * (A + A.T)
    SKEWSYM = .5 * (A - A.T)
    S = SYM.pow(2).sum() / A.pow(2).sum()
    N = SKEWSYM.pow(2).sum() / A.pow(2).sum()

    return S, N

def get_score_trace(Wq, Wk):
    """ Takes the Wq and Wk matrices (size (d,d_h)), computes the matrices A, B, and C 
    with time complexity O(d d_h^2), and computes the ratio of the norms following:
    
    norm(M)**2 = Tr(M M.T) = Tr(Wq Wk.T Wk Wq.T) = Tr(Wk.T Wk Wq.T Wq) = Tr(AB) 
    norm(S)**2 = 1/2 * (Tr(M M.T) + Tr(M M)) = 1/2 * (Tr(AB) + Tr(M M)) = 
               = 1/2 * (Tr(AB) + Tr(Wq Wk.T Wq Wk.T)) =
               = 1/2 * (Tr(AB) + Tr(Wk.T Wq Wk.T Wq) =
               = 1/2 * (Tr(AB) + Tr(CC)) 
    norm(N)**2 = 1/2 * (Tr(AB) - Tr(CC)) 

    and thus,

    norm(S)**2 / norm(M)**2 = 1/2 * (1 + Tr(CC) / Tr(AB)) 
    norm(N)**2 / norm(M)**2 = 1/2 * (1 - Tr(CC) / Tr(AB)) 

    where:

    A = Wq.T @ Wq
    B = Wk.T @ Wk
    C = Wk.T @ Wq
            
    Args:
        - Wq (numpy.ndarray): numpy matrix.
        - Wk (numpy.ndarray): numpy matrix.
    Returns:
        - tuple: Symmetric (S) and skew-symmetric (N) scores.
    """
     
    A = Wq.T @ Wq
    B = Wk.T @ Wk
    C = Wk.T @ Wq
    S = .5 * (1 + (np.einsum('ij,ji->', C, C)) / np.einsum('ij,ji->', A, B))
    N = .5 * (1 - (np.einsum('ij,ji->', C, C)) / np.einsum('ij,ji->', A, B))

    return S, N

def get_score_trace_heads(Wq, Wk):
    """ Takes the Wq and Wk matrices (size (h,d,d_h)), computes the matrices A, B, and C 
    with time complexity O(d d_h^2), and computes the ratio of the norms as above. This
    computation is done per head.
            
    Args:
        - Wq (numpy.ndarray): numpy matrix.
        - Wk (numpy.ndarray): numpy matrix.
    Returns:
        - tuple: Symmetric (S) and skew-symmetric (N) scores.
    """
     
    A = torch.matmul(Wq.transpose(-1,-2), Wq)
    B = torch.matmul(Wk.transpose(-1,-2), Wk)
    C = torch.matmul(Wk.transpose(-1,-2), Wq)
    S = .5 * (1 + (np.einsum('hij,hji->h', C, C)) / np.einsum('hij,hji->h', A, B))
    N = .5 * (1 - (np.einsum('hij,hji->h', C, C)) / np.einsum('hij,hji->h', A, B))

    return S, N

####

def get_non_random_component(Wq, Wk):
    """ Takes the Wq and Wk matrices (size (d,d_h)), computes the matrix C 
    with time complexity O(d d_h^2), and computes the non-random component of
    the matrix M following:

    Tr(M M) = Tr(Wq Wk.T Wq Wk.T)) = Tr(Wk.T Wq Wk.T Wq) = Tr(CC)

    where

    C = Wk.T @ Wq

    Args:
        - Wq (numpy.ndarray): numpy matrix.
        - Wk (numpy.ndarray): numpy matrix.
    Returns:
        - zeta (float): magnitude of non-random part of the matrix M
    """

    C = Wk.T @ Wq
    zeta = np.einsum('ij,ji->', C, C)

    return zeta