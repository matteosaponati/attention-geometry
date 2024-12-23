import numpy as np
import torch
import json
import os
import pickle
from huggingface_hub import hf_hub_download, HfApi
from safetensors import safe_open

def detect_shards(model_name: str) -> tuple:
    """
    detects the shard format and index file for the model in its hugging face repository.

    args:
        model (str): name of the model repository
    returns:
        tuple: (files, shard_format, index_file_name) where:
            - files (list): all files in the model repository
            - shard_format (str): format of the shard (e.g., "safetensors", "bin", "safetensors-noindex")
            - index_file_name (str or None): index file name if present, else None
    raises:
        ValueError: if no recognized shard format is found
    """

    files = HfApi().list_repo_files(model_name)
    index_candidates = {"safetensors": "model.safetensors.index.json",
                        "bin": "pytorch_model.bin.index.json",}
    shard_format, index_file_name = next(((fmt, idx) for fmt, idx in index_candidates.items() 
                                          if idx in files), (None, None))
    if not shard_format:
        shard_format = next(
            (fmt for fmt, ext in {"safetensors-noindex": ".safetensors", "bin-noindex": ".bin"}.items()
             if any(f.startswith("pytorch_model-") and f.endswith(ext) for f in files)), None)
        if not shard_format:
            raise ValueError("No recognized shards found.")

    return files, shard_format, index_file_name

def get_weight_map(model_name: str, index_file_name: str):
    """
    retrieves the weight map from the index file of the model.

    args:
        model (str): name of the model repository
        index_file_name (str): name of the index file containing the weight map
    returns:
        dict: the weight map extracted from the index file
    """

    weight_map = {}
    index_path = hf_hub_download(repo_id = model_name, filename=index_file_name)
    with open(index_path, "r") as f:
        index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})

    return weight_map

def get_param(model_name: str, files, shard_format, index_file_name,
               param_dir: str) -> torch.Tensor:
    """
    retrieves the parameter tensor or value from a model's shard file based on its format

    args:
        model (str): name or identifier of the model repository
        files (list): list of files in the model repository
        shard_format (str): format of the shard (e.g., "bin", "safetensors", "safetensors-noindex")
        index_file_name (str): name of the index file containing weight mappings, if applicable
        param_dir (str): parameter directory or key to look up
    returns:
        torch.Tensor or None: the parameter tensor or value, or None if not found
    """

    if shard_format in ["bin", "safetensors"]:
        weight_map = get_weight_map(model_name, index_file_name)
        shard = weight_map.get(param_dir)
        if shard is None:
            return None
        shard_path = hf_hub_download(model_name, shard)
        if shard_format == "bin":
            data = torch.load(shard_path, map_location="cpu")
            return data.get(param_dir)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            return f.get_tensor(param_dir) if param_dir in f.keys() else None
        
    else:

        if shard_format == "safetensors-noindex": ext =  ".safetensors"
        else: ext =  ".bin"
        shards = [f for f in files if f.startswith("pytorch_model-") and f.endswith(ext)]
        for i, s in enumerate(shards):
            shard_path = hf_hub_download(model_name, s)
            if ext == ".bin":
                data = torch.load(shard_path, map_location="cpu")
                if param_dir in data:
                    return data[param_dir]
            else:
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    if param_dir in f.keys():
                        return f.get_tensor(param_dir)
        return None

def get_M_shard(model_name: str, config, path: str, layer: int, 
                attn_type: str = 'BERT'):
    """
    computes the matrix product of query and key weight matrices for a given attention layer.

    args:
        model (str): name or identifier of the model repository
        config: model configuration containing hidden size and number of attention heads
        path (str): base path for locating parameters in the model
        layer (int): the attention layer number
        attn_type (str, optional): type of attention mechanism (default: 'BERT')
    returns:
        torch.Tensor: matrix product of Wq and Wk for the specified layer and attention type
    """
    
    d = config.hidden_size
    dh = config.hidden_size // config.num_attention_heads
    files, shard_format, index_file_name = detect_shards(model_name)

    if attn_type == 'GPT':        
        W_path = model_name, path[0] + f"{layer}" + path[1]
        W = get_param(model_name, files, shard_format, index_file_name, W_path)
        Wq = W[:,  : d].detach()
        Wk = W[:, d : 2*d].detach()

    elif attn_type == 'gpt-neox':
        W_path = model_name, path[0] + f"{layer}" + path[1]
        W = get_param(model_name, files, shard_format, index_file_name, W_path)
        Wq = W[ : d, :].detach()
        Wk = W[d : 2*d, :].detach()

    elif attn_type == 'grouped-attention':
        Wq_path = path[0] + f"{layer}" + path[1]
        Wk_path = path[0] + f"{layer}" + path[2]
        Wq = get_param(model_name, files, shard_format, index_file_name, Wq_path).T.detach()
        Wk = get_param(model_name, files, shard_format, index_file_name, Wk_path).T.detach()
        Wk = Wk.view(Wk.shape[0], dh, Wk.shape[1] // dh)
        repeat_factor = (Wq.shape[0] // dh) // Wk.shape[-1]
        Wk= Wk.repeat_interleave(repeat_factor, dim = 0) 
        Wk = Wk.view(Wq.shape[0], Wq.shape[0])

    else:
        Wq_path = path[0] + f"{layer}" + path[1]
        Wk_path = path[0] + f"{layer}" + path[2]
        Wq = get_param(model_name, files, shard_format, index_file_name, Wq_path).T.detach()
        Wk = get_param(model_name, files, shard_format, index_file_name, Wk_path).T.detach()
    
    return Wq.float() @ Wk.float().T

def get_M_fullmodel(model, config, path: str, layer: int, 
                attn_type: str = 'BERT'):
        
    d = config.hidden_size
    dh = config.hidden_size // config.num_attention_heads

    if attn_type == 'GPT':
        Wq = get_nested_attr(model, path[0] + f"{layer}" + path[1])[:,  : d].detach()
        Wk = get_nested_attr(model, path[0] + f"{layer}" + path[1])[:, d : 2*d].detach()
    elif attn_type == 'gpt-neox':
        Wq = get_nested_attr(model, path[0] + f"{layer}" + path[1])[ : d, :].detach()
        Wk = get_nested_attr(model, path[0] + f"{layer}" + path[1])[d : 2*d, :].detach()
    elif attn_type == 'grouped-attention':
        Wq = get_nested_attr(model, path[0] + f"{layer}" + path[1]).T.detach()
        Wk = get_nested_attr(model, path[0] + f"{layer}" + path[2]).T.detach()
        Wk = Wk.view(Wk.shape[0], dh, Wk.shape[1] // dh)
        repeat_factor = (Wq.shape[0] // dh) // Wk.shape[-1]
        Wk= Wk.repeat_interleave(repeat_factor, dim = 0) 
        Wk = Wk.view(Wq.shape[0], Wq.shape[0])
    else:
        Wq = get_nested_attr(model, path[0] + f"{layer}" + path[1]).T.detach()
        Wk = get_nested_attr(model, path[0] + f"{layer}" + path[2]).T.detach()

    return Wq @ Wk.T

def get_M_checkpoints(model: str, path: str, layer: int):

    Wq = get_nested_attr(model, path[0] + f"{layer}" + path[1]).T.detach()
    Wk = get_nested_attr(model, path[0] + f"{layer}" + path[2]).T.detach()

    return Wq @ Wk.T

def get_nested_attr(obj, attr_path):
    """
    retrieves a nested attribute or indexed value from an object based on a dot-separated path.

    args:
        obj: the object to retrieve attributes from
        attr_path (str): dot-separated path specifying the attribute to retrieve, with optional indexing
    returns:
        any: the value of the nested attribute or indexed value
    """ 

    attrs = attr_path.split('.')
    
    for attr in attrs:
        if '[' in attr and ']' in attr:
            attr_name, index = attr[: -1].split('[')
            obj = getattr(obj, attr_name)[int(index)]
        else:
            obj = getattr(obj, attr)
    
    return obj

def create_dict(model_family: str, model_name: str):
    """
    creates or retrieves a dictionary for a specified model family and model name, storing it in a specific directory structure

    parameters:
    model_family: string representing the family of the model (e.g., 'transformer', 'bert')
    model_name: string representing the specific name of the model

    returns:
    tuple containing:
    - models: dictionary loaded from the file if it exists, or an empty dictionary if the file does not exist
    - dir: string path to the directory where the model dictionary is stored
    """

    current_dir = os.getcwd()
    while not os.path.isdir(os.path.join(current_dir, "attention-geometry")):
        current_dir = os.path.abspath(os.path.join(current_dir, "../"))
        if current_dir == "/":
            raise FileNotFoundError("The directory 'attention-geometry' could not be found.")
    os.makedirs(os.path.join(current_dir, 
                             'attention-geometry/_results', 
                             model_family), exist_ok = True)
    
    dir = os.path.join(current_dir, 'attention-geometry', '_results', model_family, model_name)
    if os.path.isfile(dir):
        with open(dir, 'rb') as file:
            models = pickle.load(file)
    else: models = {}

    return models, dir

def list_all_directories(path: str):
    """
    Takes a path as input, goes through all directories inside that path,
    and returns a list of strings where each element is the name of every directory in the path.

    Parameters:
        path (str): The path to traverse.

    Returns:
        list: A sorted list of directory names in natural (human-readable) order.
    """
    def natural_sort_key(s):
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

    directory_names = []

    for root, dirs, files in os.walk(path):
        directory_names.extend(dirs)

    return sorted(directory_names, key = natural_sort_key)

def get_dir(name: str):
    """
    Navigates backward from the current directory until the "attention-geometry" directory is found,
    then constructs and returns the path to the "attention-geometry/_data/custom-models/{name}/training_output/" directory.

    Parameters:
        name (str): The name of the custom model directory.

    Returns:
        str: The path to the training output directory.

    Raises:
        FileNotFoundError: If the "attention-geometry" directory is not found.
    """
    current_dir = os.getcwd()
    while not os.path.isdir(os.path.join(current_dir, "attention-geometry")):
        current_dir = os.path.abspath(os.path.join(current_dir, "../"))
        if current_dir == "/":
            raise FileNotFoundError("The directory 'attention-geometry' could not be found.")
    dir = os.path.join(current_dir, "attention-geometry", "_data", "custom-models", name, "training_output")
    os.makedirs(dir, exist_ok=True)

    return dir

### maybe all of this is not needed anymore 
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