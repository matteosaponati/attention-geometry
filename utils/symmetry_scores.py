import numpy as np 
from .funs import get_nested_attr

def get_scores(d: int, l: int, h: int, dh: int, 
               model, path: list, 
               mode: str = "trace", model_type: str = "BERT") -> np.ndarray:
    
    """ Computes the symmetry scores for a given model and returns a 
    nnumpy array of dimension (# layers, # heads)
    """

    score_List = np.zeros((l,h))

    for i in range(l):
        
        if model_type == 'GPT':
            Wq = get_nested_attr(model, path[0] + f"{i}" + path[1])[:,  : d].view(d, h, dh).detach().numpy()
            Wk = get_nested_attr(model, path[0] + f"{i}" + path[1])[:, d : 2*d].view(d, h, dh).detach().numpy()
            
        else:
            Wq = get_nested_attr(model, path[0] + f"{i}" + path[1]).T.view(d, h, dh).detach().numpy()
            Wk = get_nested_attr(model, path[0] + f"{i}" + path[2]).T.view(d, h, dh).detach().numpy()

        for j in range(h):

            if mode == 'trace': score_List[i,j] = get_scores_trace(Wq[:, j, :], Wk[:, j, :])
            if mode == 'norm': score_List[i,j] = get_scores_norm(Wq[:, j, :] @ Wk[:, j, :].T)
            if mode == 'sum': score_List[i,j] = get_scores_sum(Wq[:, j, :] @ Wk[:, j, :].T)

    return  score_List

def get_scores_full(d: int, l: int, h: int, dh: int, 
               model, path: list, 
               mode: str = "trace", model_type: str = "BERT") -> np.ndarray:
    
    """ Computes the symmetry scores for a given model and returns a 
    nnumpy array of dimension (# layers, # heads)
    """

    score_List = np.zeros((l,h))

    for i in range(l):
        
        if model_type == 'GPT':
            Wq = get_nested_attr(model, path[0] + f"{i}" + path[1])[:,  : d].detach().numpy()
            Wk = get_nested_attr(model, path[0] + f"{i}" + path[1])[:, d : 2*d].detach().numpy()
            
        else:
            Wq = get_nested_attr(model, path[0] + f"{i}" + path[1]).T.detach().numpy()
            Wk = get_nested_attr(model, path[0] + f"{i}" + path[2]).T.detach().numpy()

        for j in range(h):

            if mode == 'trace': score_List[i,j] = get_scores_trace(Wq, Wk)
            if mode == 'norm': score_List[i,j] = get_scores_norm(Wq @ Wk.T)
            if mode == 'sum': score_List[i,j] = get_scores_sum(Wq @ Wk.T)

    return  score_List

def get_scores_trace(Wq, Wk):
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

    score = .5 * (1 + (np.einsum('ij,ji->', C, C) / np.einsum('ij,ji->', A, B)))

    return score

def get_scores_norm(A):
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

def get_scores_sum(A):
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