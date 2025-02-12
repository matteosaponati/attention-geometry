# attention-geometry


# Training Custom Models

*Please note that this is a cleanup version from the repository used for the experiments in the paper. Therefore, the results might slightly differ. If you notice bugs or want to reproduce the results exactly as reported in the paper, please get in touch with sage@zhaw.ch (he can give you access to the original code). If you are interested in a detail log of the training metrics, please have a look at https://wandb.ai/sagerpascal/symmetric-attention-from-scratch-final-2.*


## Environment Variables

Please create a `.env` file in the root directory (from where you start the code) with the following content:

```bash
HF_TOKEN=<your-huggingface-token>
WANDB_API_KEY=<your_api_key>
```

## Install dependencies

Please install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```


## Data

We use three datasets for training our models:

- Jigsaw Toxic Comment Classification Challenge
- RedPajama
- Wikipedia

While we access the RedPajama and Wikipedia dataset directly from Huggingface, the Jigsaw dataset has to be downloaded from Kaggle:

- https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview

After downloading the dataset, please extract the files and then run the following script:

```
python prepare_jigsaw.py --raw <path-to-raw-data> --train <path-to-train-folder> --test <path-to-test-folder>
```

The other dataset do not require any additional preparation.

## Training

You can run the training script directly:

```bash
python train.py --train <dataset> --model-save <save-path> --model_name <model-name> --mode <mode> --init <init>
```

- `--train` please set this to either `<jigsaw-data-path>`, `red_pajama`, or `wiki` to train the model on the respective dataset.
- `--model-save` the path where the model should be saved.
- `--model_name` the name of the model to be trained; either 'bert_small' or 'bert'
- `--mode` the mode to train the model in; either 'encoder' or 'decoder'
- `--init` the initialization method for the model; set optionally to 'symmetric-init'

For example, run the following command to train a model on the Wikipedia dataset:

```bash
python train.py --train wiki --model-save models/wiki --model_name bert_small --mode encoder --init symmetric-init
```


