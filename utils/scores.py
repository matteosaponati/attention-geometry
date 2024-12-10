import numpy as np 
import torch
import sys
from .funs import get_M_shard, get_M_fullmodel

def get_scores(models: dict,
               model_name: str, model, config, path: list,
               download_model: bool = True,
               attn_type: str = "BERT") -> np.ndarray:
    """
    calculates symmetry and directionality scores for all layers of a model and updates a dictionary with the results

    parameters:
        models: dictionary to store the calculated scores and model configuration
        model_name: string representing the name of the model
        model: the model object to analyze
        config: configuration object with model parameters such as the number of hidden layers
        path: list specifying the nested attribute path to the layer weights
        download_model: boolean indicating whether to calculate scores from a full model or a shard, default is True
        attn_type: string specifying the attention type ('BERT', 'GPT', etc.), default is 'BERT'

    returns:
        updated models dictionary with added scores and configuration for the specified model
    """
    
    parameters = model.num_parameters()

    l = config.num_hidden_layers
    scores_symmetry = np.zeros(l)
    scores_directionality = np.zeros(l)

    for layer in range(l):
        
        progress = (layer + 1) / l * 100
        sys.stdout.write(f"\r{int(progress):3}% processing layer {layer + 1}/{l}")
        sys.stdout.flush()
        
        if download_model == True: M = get_M_fullmodel(model, config, path, layer, attn_type)
        else: M = get_M_shard(model_name, config, path, layer, attn_type)
        scores_symmetry[layer] = symmetry_score(M)
        scores_directionality[layer] = directionality_score(M)

    models[model_name] = [config, scores_symmetry, scores_directionality, [parameters, parameters]]
    
    sys.stdout.write("\r100% processing complete \n")
    sys.stdout.flush()

    return models

def symmetry_score(A):
    """ Takes a square matrix A, computes its symmetric component SYM (time complexity O(d)),
    computes the square norm of SYM and A (sum of the square of matrix entries), 
    and computes the ratio of the norms.
    """
    SYM = .5 * (A + A.T)
    score = (SYM ** 2).sum() / (A ** 2).sum() 
    return score

def directionality_score(A, num_std: int = 2):
    """ Takes a square matrix A, computes its symmetric component SYM (time complexity O(d)),
    computes the square norm of SYM and A (sum of the square of matrix entries), 
    and computes the ratio of the norms.
    """

    row_norms = torch.norm(A, dim=1)  # row norms
    column_norms = torch.norm(A, dim=0)  # column norms

    row_threshold = row_norms.mean().item() + num_std * row_norms.std().item()
    col_threshold = column_norms.mean().item() + num_std * column_norms.std().item()
    # row_outliers = np.sum(row_norms > row_threshold)
    # col_outliers = np.sum(column_norms > col_threshold)

    row_excess = torch.sum((row_norms[row_norms > row_threshold] - row_threshold))
    col_excess = torch.sum((column_norms[column_norms > col_threshold] - col_threshold))
    
    denom_excess = row_excess + col_excess
    if denom_excess == 0: score = 0.0
    else: score = (col_excess - row_excess) / denom_excess
    
    # denom = row_outliers + col_outliers
    # if denom == 0: score = 0.0
    # else: score = (row_outliers - col_outliers) / denom
    
    return score