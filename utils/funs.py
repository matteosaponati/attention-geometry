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
    
def get_interpolation(data):

    data = data.to_numpy()
    x = np.arange(data.size)
    data = np.interp(x, x[~np.isnan(data)], 
                                  data[~np.isnan(data)])

    return data

def get_symmetry_training(data, layers, heads, epochs):
    
    scores = np.zeros((layers, heads, epochs))
    for l in range(layers):
        for h in range(heads):
            scores[l, h, :] = np.array(get_interpolation(data[f'Layer {l}/Head {h} Symmetry']))
    
    return scores

def get_symmetry_layers_heads(data, layers, heads):
    return [(get_interpolation(data[f'Layer {l}/Head {h} Symmetry']))[-1] for l in range(layers) for h in range(heads)]

def get_specs(data):
    return [get_interpolation(data['_step']),
            get_interpolation(data['train/global_step']),
            get_interpolation(data['Mean Symmetry']),
            get_interpolation(data['Min Symmetry']),
            get_interpolation(data['Max Symmetry']),
            get_interpolation(data['Median Symmetry']),
            get_interpolation(data['Variance Symmetry']),
            get_interpolation(data['train/loss'])]