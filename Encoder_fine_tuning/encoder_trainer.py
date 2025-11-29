import os
import csv
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    BertForTokenClassification,
    RobertaTokenizerFast,#RobertaTokenizer,
    # RobertaModel0, #This is from the official docs, but haven't found a version for token classification so leaving it for now
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    TrainerCallback,
    AutoTokenizer,
    set_seed,
)
from transformers.modeling_utils import PreTrainedModel
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import shutil
import pandas as pd

def get_model_by_name(model_name: str, label2id: Dict[str, int], id2label: Dict[int, str], hidden_dropout_prob: float=0.5,
                      attention_probs_dropout_prob: float=0.5) -> Union[BertForTokenClassification, AutoModelForTokenClassification]:
    if model_name in ["FacebookAI/xlm-roberta-base", "FacebookAI/roberta-base"] or "japanese" in model_name:
        return AutoModelForTokenClassification.from_pretrained(model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,)
    else:
        return BertForTokenClassification.from_pretrained(model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )

def get_tokenizer_by_model_name(model_name: str) -> Union[AutoTokenizer, RobertaTokenizerFast]:
    if model_name == "FacebookAI/roberta-base":
        return RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
    elif "japanese" in model_name:
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)
    else:
        return AutoTokenizer.from_pretrained(model_name)


def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(-1)

    true_labels, pred_labels = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            true_labels.append(id2label[label_id])
            pred_labels.append(id2label[pred_id])

    return {
        "eval_f1": f1_score(true_labels, pred_labels, average="macro"),
        "precision": precision_score(true_labels, pred_labels, average="macro"),
        "recall": recall_score(true_labels, pred_labels, average="macro"),
        "accuracy": accuracy_score(true_labels, pred_labels),
    }


def encode_labels(example):
    example["labels"] = [label2id[tag] for tag in example["tags"]]
    return example


def tokenize_and_align_labels(example):
    tokenized = tokenizer(example["tokens"], is_split_into_words=True, truncation=True)
    word_ids = tokenized.word_ids()
    previous_word_idx = None
    labels = []
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(example["labels"][word_idx])
        else:
            labels.append(-100)
        previous_word_idx = word_idx
    tokenized["labels"] = labels
    return tokenized


def tokenize_and_align_labels_slow(example):
    # For slow tokenizers, we need to align manually
    # Tokenize each word and assign the label to all resulting tokens
    tokens = example["tokens"]
    labels = example["labels"]
    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,  # For slow tokenizer, this works as expected
        truncation=True,
        padding='max_length',
        max_length=128,  # or whatever length you need
    )
    # Manually align labels
    word_ids = []
    cur = 0
    for token in tokenized_inputs["input_ids"]:
        # For slow tokenizers, you can align using the tokens list
        # Here, we assign the label of the current word to all its sub-tokens
        # This is a simplification; you may need to adjust for special tokens
        if cur < len(labels):
            word_ids.append(labels[cur])
            cur += 1
        else:
            word_ids.append(-100)  # Padding label for special tokens
    tokenized_inputs["labels"] = word_ids[:len(tokenized_inputs["input_ids"])]
    return tokenized_inputs


def get_tokenized_dataset(model_name: str, dataset: DatasetDict) -> DatasetDict:
    if "japanese" in model_name:
        return dataset.map(tokenize_and_align_labels_slow, batched=False)
    else:
        return dataset.map(tokenize_and_align_labels)

        
@dataclass
class EncoderTrainer:
    model_name: str
    output_dir: Path = None
    seed: int = -1
    train_data: pd.DataFrame = None
    test_data: pd.DataFrame = None
    trained_model: PreTrainedModel = None #For type, see: https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Trainer.model

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        global tokenizer #TODO - fix this so that everything will be inside the class, including all the functions that need the tokenizer (e.g. tokenize_and_align_labels).
        tokenizer = get_tokenizer_by_model_name(self.model_name)


    def delete_checkpoints(self):
        output_path = Path(self.output_dir)
        for folder in output_path.glob("checkpoint-*"):
            if folder.is_dir():
                print(f"Deleting {folder}")
                shutil.rmtree(folder)


    def train(self, should_delete_checkpoints: bool, final_trained_model_dir: Path = None, custom_train_args: Dict[str, Union[float, str, bool]] = {}):
        global label2id, id2label, tokenizer  # used in encode_labels and compute_metrics

        if len(custom_train_args)>0:
            print(f"{datetime.now()} Got custom training args {custom_train_args}")
        set_seed(self.seed)
        self.train_data = self.train_data.sample(frac=1, random_state=self.seed).reset_index(drop=True) #Shuffling order of rows
        train_dataset = Dataset.from_pandas(self.train_data[["tokens", "tags"]])
        test_dataset = Dataset.from_pandas(self.test_data[["tokens", "tags"]])
        dataset = DatasetDict({
            "train": train_dataset,
            "test": test_dataset,
        })

        unique_tags = sorted(set(tag for row in dataset["train"]["tags"] for tag in row))
        label2id = {tag: i for i, tag in enumerate(unique_tags)}
        id2label = {i: tag for tag, i in label2id.items()}

        dataset = dataset.map(encode_labels)
        
        tokenized_dataset = get_tokenized_dataset(self.model_name, dataset) #tokenized_dataset = dataset.map(tokenize_and_align_labels)

        model = get_model_by_name(self.model_name, label2id, id2label)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=20,
            weight_decay=0.01,
            warmup_steps=0,
            logging_dir="./logs",
            logging_strategy="epoch",
            logging_steps=1000,
            save_total_limit=2,
            seed=self.seed,
            dataloader_num_workers=4,
            disable_tqdm=False,
            report_to=[],
            gradient_accumulation_steps=4,
            gradient_checkpointing=False,
            fp16=True,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
        )

        for k, v in custom_train_args.items():
            setattr(training_args, k, v)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            processing_class=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer),
            compute_metrics=compute_metrics
        )

        trainer.train()
        if should_delete_checkpoints:
            try:
                self.delete_checkpoints()
            except Exception as e:
                print(f"{datetime.now()} An exception occurred while trying to delete checkpoints.\nException: {e}\n{traceback.format_exc()}")
        
        if final_trained_model_dir is not None:
            try:
                print(f"{datetime.now()} Saving trained model to {final_trained_model_dir}")
                final_trained_model_dir.mkdir(parents=True, exist_ok=True)
                trainer.save_model(final_trained_model_dir)
            except Exception as e:
                print(f"{datetime.now()} An exception occurred while trying to save model, unable to save it.\nException: {e}\n{traceback.format_exc()}")
        
        self.trained_model = trainer.model

    def test(self, test_data: Dataset, return_metrics: Optional[List[str]] = None) -> Dict[str, float]:
        assert self.trained_model is not None, "No model exists, so can't run test(). Call train() first or set it manually"
        
        global label2id, id2label, tokenizer
        test_dataset = Dataset.from_pandas(test_data[["tokens", "tags"]])

        unique_tags = sorted(set(tag for row in test_dataset["tags"] for tag in row))
        label2id = {tag: i for i, tag in enumerate(unique_tags)}
        id2label = {i: tag for tag, i in label2id.items()}
        test_dataset = test_dataset.map(encode_labels)
        tokenized_dataset = get_tokenized_dataset(self.model_name, test_dataset) #tokenized_test = test_dataset.map(tokenize_and_align_labels)

        # Set up trainer with dummy TrainingArguments (no training will be done)
        args = TrainingArguments(
            output_dir=self.output_dir,  # temporary folder
            per_device_eval_batch_size=32,
            report_to=[],
            do_train=False,
            do_eval=True,
            disable_tqdm=True,
        )

        trainer = Trainer(
            model=self.trained_model,
            args=args,
            processing_class=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer),
            compute_metrics=compute_metrics,
        )

        results = trainer.evaluate(eval_dataset=tokenized_dataset)
        if return_metrics:
            return {k: v for k, v in results.items() if k in return_metrics}
        return results