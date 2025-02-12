#
<h1 align="center">The underlying structures of self-attention</h1>

This is a repository for the paper:
<br/><br/>
"*The underlying structures of self-attention: symmetry, directionality, and emergent dynamics in Transformer training*", M Saponati, P Sager, PV Aceituno, T Staldemann, B Grewe. 
ArXiv (2025). <br/>
[link to the arxiv]

## Table of Contents

1. [Installation](#Installation)
2. [Structure](#Structure)
3. [Training custom models](#Training-custom-models)
4. [Citation](#citation)
5.  [License](#license)

-------------------------

# Installation

The current version of the scripts has been tested with Python 3.12.3 and Pytorch 2.5.1. All the dependencies are listed in the environment.yaml file. 
The project has a pip-installable package. How to set it up:

- `git clone` the repository 
- `pip install -e . `

# Structure

This repo is structured as follows:

+ `./custom-training-models/` contains the code necessary to train the custom models (see below for a step-by-step guide).
+ `./notebooks-figures-tables/` contains the Jupyter notebooks to reproduce the figures and tables of the paper.
+ `./notebook-scores/` contains the Jupyter notebooks to calculate and save the directionality and symmetry scores for every model we considered.
+ `./utils/` contains the Python modules for downloading the pretrained models from Huggingface, calculating the directionality and symmetry score, and visualizing the results.

+ `environment.yaml` configuration file with all the dependencies listed
+ `setup.py` python script for installation with pip

# Training-custom-models

*Please note that this is a cleanup version from the repository used for the experiments in the paper. Therefore, the results might slightly differ. If you notice bugs or want to reproduce the results exactly as reported in the paper, please get in touch with sage@zhaw.ch. If you are interested in a detail log of the training metrics, please have a look at [wandb](https://wandb.ai/sagerpascal/attention-geometry/).*


### Environment Variables

Create an `.env` file in the root directory (from where you start the code) with the following content:

```bash
HF_TOKEN=<your-huggingface-token>
WANDB_API_KEY=<your_api_key>
```

### Install dependencies

Iinstall the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```


### Data

We use three datasets for training our models:

- Jigsaw Toxic Comment Classification Challenge
- RedPajama
- Wikipedia

While we access the RedPajama and Wikipedia dataset directly from Huggingface, the Jigsaw dataset has to be downloaded from Kaggle:

- https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview

After downloading the dataset, extract the files and then run the following script:

```
python prepare_jigsaw.py --raw <path-to-raw-data> --train <path-to-train-folder> --test <path-to-test-folder>
```

The other dataset do not require any additional preparation.

### Training

You can run the training script directly:

```bash
python train.py --train <dataset> --model-save <save-path> --model_name <model-name> --mode <mode> --init <init>
```

- `--train` set this to either `<jigsaw-data-path>`, `red_pajama`, or `wiki` to train the model on the respective dataset.
- `--model-save` the path where the model should be saved.
- `--model_name` the name of the model to be trained; either 'bert_small' or 'bert'
- `--mode` the mode to train the model in; either 'encoder' or 'decoder'
- `--init` the initialization method for the model; set optionally to 'symmetric-init'

For example, run the following command to train a model on the Wikipedia dataset:

```bash
python train.py --train wiki --model-save models/wiki --model_name bert_small --mode encoder --init symmetric-init
```


# Citation
citation.<br/>
```
@article{
}
```


# License

MIT License

Copyright (c) 2025 Matteo Saponati

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.