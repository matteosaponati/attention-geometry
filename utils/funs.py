import numpy as np

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