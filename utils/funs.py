import numpy as np 

def scores(A):
    """
    Given a square matrix A, calculate the symmetric (S) and skew-symmetric (N) scores of a matrix.
    Args:
        - A (numpy.ndarray) : square numpy matrix.

    Returns:
        - tuple : Symmetric (S) and skew-symmetric (N) scores.
    """
     
    S = np.linalg.norm(.5 * (A + A.T), 'fro') / np.linalg.norm(A, 'fro')
    N = np.linalg.norm(.5 * (A - A.T), 'fro') / np.linalg.norm(A, 'fro')

    return S, N

def dotproduct_normal(A,rep=100):
    """
    Given a square matrix A, calculate the symmetric (S) and skew-symmetric (N) scores of a matrix.
    Args:
        - A (numpy.ndarray) : square numpy matrix.

    Returns:
        - tuple : Symmetric (S) and skew-symmetric (N) scores.
    """

    u = np.random.randn(rep,A.shape[0])
    v = np.random.randn(rep,A.shape[0]) 

    Dots = u @ A @ v.T
    DotsCanonical = u @ v.T

    return Dots, DotsCanonical