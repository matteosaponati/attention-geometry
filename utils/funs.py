import numpy as np
import torch.nn as nn

def get_nested_attr(obj, attr_path): 
    """ Recursively get nested attributes"""

    attrs = attr_path.split('.')
    
    for attr in attrs:
        if '[' in attr and ']' in attr:
            attr_name, index = attr[: -1].split('[')
            obj = getattr(obj, attr_name)[int(index)]
        else:
            obj = getattr(obj, attr)
    
    return obj

def count_outliers(arr):
    
    # Calculate the first (Q1) and third (Q3) quartiles
    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)
    
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Count the number of outliers
    outliers = arr[((arr < Q1 - 1.5 * IQR) | (arr > Q3 + 1.5 * IQR))]
    
    return outliers

def count_parameters(model, Embedding_FLAG = False):
    
    if Embedding_FLAG == True:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    else:
        return sum(p.numel() for name, module in model.named_modules()
        for p in module.parameters(recurse=False) if p.requires_grad and not isinstance(module, nn.Embedding))