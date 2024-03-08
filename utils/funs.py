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

    u = np.random.rand(rep,A.shape[0])
    v = np.random.rand(rep,A.shape[0]) 

    dot = np.zeros(rep)
    dotC = np.zeros(rep)

    for k in range(rep):
        dot[k] = np.dot(u[k,:], A @ v[k,:].T) / (np.linalg.norm(u[k,:]) * np.linalg.norm(A @ v[k,:].T))        
        dotC[k] = np.dot(u[k,:],v[k,:]) / (np.linalg.norm(u[k,:]) * np.linalg.norm(v[k,:]))

    return dot,dotC

def isotropic_normal(A,rep=100):
    """
    Given a square matrix A, calculate the symmetric (S) and skew-symmetric (N) scores of a matrix.
    Args:
        - A (numpy.ndarray) : square numpy matrix.

    Returns:
        - tuple : Symmetric (S) and skew-symmetric (N) scores.
    """

    u = np.random.rand(rep,A.shape[0])
    dot = np.zeros(rep)
    dotC = np.zeros(rep)
    
    for k in range(rep):
        dot[k] = np.dot(u[k,:], A @ u[k,:].T) / (np.linalg.norm(u[k,:]) * np.linalg.norm(A @ u[k,:].T))        
        dotC[k] = np.dot(u[k,:],u[k,:]) / (np.linalg.norm(u[k,:])**2)

    return dot,dotC