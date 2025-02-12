# Quantification of directionality and symmetry scores

1. Import the models and config required from the `transformers` package.
</br>
Example: </br>`from transformers import BertModel, AutoConfig`

2. Define the directory for saving the results and the name of the dictionary, and get both dictionary and directory as output.
</br>
Example: </br>`models, dir = create_dict('language-models', 'BERT.pkl')` 

3. Define the `path` where the first element is the path to the self-attention model's attribute, and the other two are the path to the Wq and Wk parameters of the `nn.Linear` method. Note that different models might have different paths to the Wq and Wk matrices.
</br>
Example: </br> `path = ["encoder.layer[", 
        "].attention.self.query.weight", 
        "].attention.self.key.weight"]`
</br>


4.  Set `model_name` to be the name of the model as from Huggignface, get the config file, download the model from HuggingFace, and finally compute scores with the `get_scores` function in `utils.scores`.
</br>
Example: </br>`model_name = "google/bert_uncased_L-2_H-128_A-2"`
</br>
`config = AutoConfig.from_pretrained(model_name)`
</br>
`model = AutoModel.from_pretrained(model_name)`
</br>
`models = get_scores(models,
                    model_name, model, config,
                    path, 
                    custom_checkpoint = False, download_model = True,
                    attn_type = "BERT")`
</br></br>
The function `get_scores` takes the dictionary as input and gives it back as output with the new model as a new key: `custom_checkpoint` is True if working with custom models, `download_model` is True if you want to download the full model and compute the scores and is False if you want to compute the scores without donwloading the model (necessary for big models), `attn_type` defines how to extract Wq and Wk from the model. 

5. After computing the scores from all models in the family, save the dictionary.
</br>
Example: </br>`with open(dir, 'wb') as file:
    pickle.dump(models, file)` 