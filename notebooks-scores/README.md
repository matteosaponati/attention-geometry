

1 - import from `transformers` the models and config required, e.g.: 
</br>
`from transformers import BertModel, AutoConfig`

2 - define the directory for saving the results and the name of the dictionary, and get both dictionary and directory as output, e.g.:
</br>
`models, dir = create_dict('language-models', 'BERT.pkl')` 

3 - define the `path` where the first element is the path to the self-attention model's attribute, and the other two are the path to the Wq and Wk parameters of the nn.Linear methods, e.g.:
</br>
`path = ["encoder.layer[", 
        "].attention.self.query.weight", 
        "].attention.self.key.weight"]`
</br>
(usually GPT models define query, key, and value together in one nn.Linear, `path = ["h[", "].attn.c_attn.weight"]`)

4 - define `model_name` (you get it from HuggingFace), get the config with `AutoConfig`,  get model from HuggingFace and compute scores with `get_scores`, e.g.:
`model_name = "google/bert_uncased_L-2_H-128_A-2"`
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

The function `get_scores` takes the dictionary as input and gives it back as output with the new model as a new key: `custom_checkpoint` is True if working with custom models, `download_model` is True if you want to download the full model and compute the scores, is False if you want to compute the scores without donwloading the model (necessary for big models), `attn_type` defines how to extract Wq and Wk from the model. 

5 - After computing the scores from all models in the family, save the dictionary with:
</br>
`with open(dir, 'wb') as file:
    pickle.dump(models, file)` 