import gzip
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import List
from typing import Literal

import huggingface_hub
import torch
import transformers
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from custom_attention import BertSelfAttentionSymmetricInit
from wandb_callback import WandbSymmetryCallback
from torch.utils.data import Dataset
import torch.distributed as dist
from transformers import (AutoConfig, AutoTokenizer, BertForMaskedLM,
                          BertLMHeadModel, DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)
from transformers.models.bert.modeling_bert import BertSelfAttention

from dataset import Sample, TrainDataset

load_dotenv()

from datasets import IterableDataset

tmp_path = os.getenv("TMP_PATH") # "/scratch"
hf_token = os.getenv("HF_TOKEN")

assert hf_token is not None, "HF_TOKEN must be set"
assert tmp_path is not None, "TMP_PATH must be set"
assert os.getenv("WANDB_RUN_ID") is not None, "WANDB_RUN_ID must be set"
assert os.getenv("WANDB_PROJECT") is not None, "WANDB_PROJECT must be set"
assert os.getenv("WANDB_RUN_ID") is not None, "WANDB_RUN_ID must be set"

os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["WANDB_DIR"] = os.path.abspath(tmp_path)
os.environ["HF_DATASETS_CACHE"] = os.path.abspath(f"{tmp_path}/.cache/huggingface/datasets")
huggingface_hub.login(hf_token)

transformers.set_seed(0)


def recursive_setattr(obj, attr, value):
    attr = attr.split('.', 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)


def replace_layer(model, module, class_old, class_new):
    if isinstance(module, class_old):
        target_state_dict = deepcopy(module.state_dict())
        if hasattr(module, "config"):
            new_module = class_new(module.config)
        else:
            new_module = class_new(model.config)
        if not isinstance(new_module, BertSelfAttentionSymmetricInit):
            new_module.load_state_dict(target_state_dict, strict=False)
        return new_module
    else:
        return module


class TorchDataset(Dataset):

    def __init__(self, data: dict):
        self.data = data

    def __len__(self) -> int:
        return self.data['input_ids'].shape[0]

    def __getitem__(self, item: int):
        return {
            k: v[item]
            for k, v in self.data.items()
        }

    @staticmethod
    def from_samples(
            samples: List[Sample],
            tokenizer: AutoTokenizer,
            include_labels: bool = True,
            include_next_sentence_label: bool = False,
    ) -> 'TorchDataset':
        torch_data = tokenizer(
            [s.text for s in samples],
            truncation=True,
            padding=True,
            return_tensors='pt',
        )
        if include_labels:
            torch_data['labels'] = torch.LongTensor([s.label for s in samples])

        return TorchDataset(data=torch_data)


def train_from_scratch(
        model_dir: Path,
        train_data: TrainDataset,
        model_name: str,
        train_mode: Literal['encoder', 'decoder'],
        device_batch_size: int,
        gradient_accumulation_steps: int,
        init_symmetric: str = None,
        load_checkpoint: str = None,
        rank: int = 0
):
    if rank == 0:
        wandb.login()

        if not model_dir.exists():
            model_dir.mkdir(parents=True)

    if model_name.split("-")[0] == "bert_small":
        config = AutoConfig.from_pretrained('prajjwal1/bert-mini')
        tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

        if train_mode == 'encoder':
            model = BertForMaskedLM(config)
            train_mode_name = train_mode
        elif train_mode == 'decoder':
            config.is_decoder = True
            train_mode_name = train_mode
            model = BertLMHeadModel(config)
        else:
            raise NotImplementedError()

    elif model_name.split("-")[0] == "bert":

        config = AutoConfig.from_pretrained('google-bert/bert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
        if train_mode == 'encoder':
            model = BertForMaskedLM(config)
            train_mode_name = train_mode
        elif train_mode == 'decoder':
            config.is_decoder = True
            train_mode_name = train_mode
            model = BertLMHeadModel(config)
        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError()

    if init_symmetric is not None:
        if init_symmetric == 'symmetric-init':
            for name, module in tuple(model.named_modules()):
                if name:
                    new_module = replace_layer(model, module, BertSelfAttention, BertSelfAttentionSymmetricInit)
                    if type(new_module) == BertSelfAttentionSymmetricInit:
                        recursive_setattr(model, name, new_module)
        else:
            raise NotImplementedError()

    if rank == 0:
        print(model)

    if tokenizer.model_max_length > 10000:
        tokenizer.model_max_length = min(tokenizer.max_model_input_sizes.values())

    if isinstance(train_data, TrainDataset):
        train = TorchDataset.from_samples(
            samples=train_data.train_samples,
            tokenizer=tokenizer,
            include_labels=True,
        )
    else:
        store_path = f"{tmp_path}/tokenized_{train_data.name}"
        train_data_texts = train_data['train']

        if isinstance(train_data_texts, IterableDataset):

            train = train_data_texts.map(
                lambda x: tokenizer([" ".join(x) for x in x["raw_content"]], truncation=True, padding='max_length', max_length=512), batched=True)

        else:
            if not Path(store_path).exists() and rank == 0:
                Path(store_path).mkdir(parents=True)
                tokenized_train_data_texts = train_data_texts.map(
                    lambda x: tokenizer([" ".join(x) for x in x["text"]], truncation=True, padding='max_length', max_length=512), batched=True, num_proc=8, )
                tokenized_train_data_texts.save_to_disk(store_path)

            dist.barrier()
            tokenized_train_data_texts = train_data_texts.load_from_disk(store_path)

            train = tokenized_train_data_texts

    mode = ""
    if init_symmetric is not None:
        mode = f"--{init_symmetric}"


    training_output = model_dir / f"{model_name}-{train_mode}-{train_data.name}{mode}" / "training_output" / os.getenv("WANDB_RUN_ID")
    log_output = model_dir / f"{model_name}-{train_mode}-{train_data.name}{mode}" / "training_logs" / os.getenv("WANDB_RUN_ID")

    if rank == 0:
        print("save training_output to", training_output)
        print("save log_output to", log_output)

        print("save training_output to", training_output)
        print("save log_output to", log_output)

        if not training_output.exists():
            training_output.mkdir(parents=True)

        if not log_output.exists():
            log_output.mkdir(parents=True)

    run_name = f"{model_name}--{train_mode_name}--{train_data.name}{mode}"
    run_id = os.getenv("WANDB_RUN_ID")


    args = TrainingArguments(
        output_dir=str(training_output),
        # logging_dir=str(log_output),
        # num_train_epochs=10 if train_data.name == "wikipedia" else 100,
        max_steps=200_000,
        per_device_train_batch_size=device_batch_size,  # or use 32
        per_device_eval_batch_size=device_batch_size,  # or use 32
        gradient_accumulation_steps=gradient_accumulation_steps,  # or use 8
        warmup_steps=200,
        weight_decay=0.01,
        learning_rate=5e-5,
        save_strategy="steps",
        save_steps=1_000,
        save_total_limit=1,  # or increase to e.g., 200 to save and store every 1000 steps
        evaluation_strategy="no",
        report_to="wandb",
        fp16=True,
        logging_steps=200,
        run_name=run_name,
        ignore_data_skip=train_data.name == "red_pajama",
    )

    if rank == 0:
        print("WARNING: run_id is", run_id, "Trying to continue WANDB run", load_checkpoint is not None)

        wandb.init(
                project=os.environ["WANDB_PROJECT"],
                name=run_name,
                id=run_id,
                resume='must' if run_id is not None and load_checkpoint is not None else 'never',
                fork_from=None,
                # resume_from=f"{run_id}?_step={step}" if run_id is not None else None,
        )


    dist.barrier()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15),
    )

    wandb_sym_callback = WandbSymmetryCallback(trainer)
    trainer.add_callback(wandb_sym_callback)

    trainer.train(resume_from_checkpoint=load_checkpoint)

    if rank == 0:
        trainer.save_model(output_dir=str(model_dir))
        tokenizer.save_pretrained(save_directory=str(model_dir))

        wandb.finish()


def main(
        train_path: str,
        model_out: Path,
        model_name: str,
        train_mode: Literal['encoder', 'decoder'],
        device_batch_size: int,
        gradient_accumulation_steps: int,
        init_symmetric: str = None,
        load_checkpoint: str = None,
        rank: int = 0
):
    if train_path.lower() == 'wiki':
        wiki_data = load_dataset("wikipedia", "20220301.en", num_proc=8)
        wiki_data.name = "wikipedia"
        train_datasets = [wiki_data]

    elif train_path.lower() == 'red_pajama':
        red_pajama_data = load_dataset("togethercomputer/RedPajama-Data-V2", languages=["en"], snapshots=["2023-14"],
                                       name="default", streaming=True)
        red_pajama_data.name = "red_pajama"
        train_datasets = [red_pajama_data]


    elif Path(train_path).exists():

        train_datasets = []
        for f in Path(train_path).glob("*.json.gz"):
            if f.name in ['jigsaw.json.gz']:  # lets just start with one dataset
                with gzip.open(f, "rt") as fin:
                    train_datasets.append(TrainDataset.from_json(json.load(fin)))

    else:
        raise ValueError(f"train_path {train_path} not found")

    for train_data in train_datasets:
        train_from_scratch(
            train_data=train_data,
            model_dir=model_out,
            model_name=model_name,
            train_mode=train_mode,
            device_batch_size=device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            init_symmetric=init_symmetric,
            load_checkpoint=load_checkpoint,
            rank=rank,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", dest="train", type=str, required=True)
    parser.add_argument(
        "--model-save", dest="model_out", type=Path, required=True)
    parser.add_argument("--model_name",
                        choices=['bert_small', 'bert'],
                        type=str,
                        default='bert_small',
                        help="the model to use"
                        )
    parser.add_argument("--mode",
                        choices=['encoder', 'decoder'],
                        type=str,
                        default='encoder',
                        help="how to train the model (either encoder or decoder)"
                        )
    parser.add_argument("--device_batch_size",
                        type=int,
                        default=32,
                        )
    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=8,
                        )
    parser.add_argument("--local-rank",
                        type=int,
                        default=0,
                        dest='rank',
                        )
    parser.add_argument("--init",
                        type=str,
                        default=None,
                        choices=[None, 'symmetric-init'],
                        help="How to initialize the model")
    parser.add_argument("--checkpoint",
                        type=str,
                        default=None,
                        help="checkpoint to load"
                        )
    args = parser.parse_args()

    main(
        train_path=args.train,
        model_out=args.model_out,
        model_name=args.model_name,
        train_mode=args.mode,
        device_batch_size=args.device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        init_symmetric=args.init,
        load_checkpoint=args.checkpoint,
        rank=args.rank
    )
