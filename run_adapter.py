#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import spacy
nlp = spacy.load('en_core_web_md')

from parent import parent
import re

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from metrics import *
from utils import *

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset

import torch

#import evaluate # don't support rouge-precision and rouge-recall

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.adapters import AdapterArguments, Seq2SeqAdapterTrainer, setup_adapter_training, AutoAdapterModel
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]

ignore_relations = ['instance of', 'sex or gender', 'country of citizenship']

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    custom_dataset_name: Optional[str] = field(
        default='wida2wl', metadata={"help": "The name of the dataset that you use by your own."}
    )
    
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    val_min_target_length: Optional[int] = field(
        default=4,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    use_adapter: Optional[str] = field(
        default='',
        metadata={
            "help": (
                "Set an adapter name to use."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    epochs: Optional[int] = field(
        default=3,
        metadata={
            "help": (
                "Number of epochs to use for training."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    patience: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Stop training when the metric specified for `metric_for_best_model` worsend for `patience` number of"
                " evaluation calls."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.test_file is None:
            raise ValueError("Need either a dataset name or a training/validation/test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, AdapterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(42) # training_args.seed

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # accelerate
    '''
    checkpoint = "facebook/opt-13b"
    model = AutoModelForCausalLM.from_pretrained(
            checkpoint, device_map="auto", offload_folder="offload", offload_state_dict = True, \
            torch_dtype=torch.float16
    )
    '''

    '''
    from accelerate import infer_auto_device_map, init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained("facebook/opt-13b")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    device_map = infer_auto_device_map(model)
    '''
    
    model_class = AutoAdapterModel if data_args.use_adapter != '' else AutoModelForSeq2SeqLM
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        device_map="auto",
        offload_folder="offload",
        offload_state_dict = True,
        #torch_dtype=torch.float16
    )

    
    '''model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )'''
    
    '''if (data_args.use_adapter != ''):
        #training_args.do_predict
        #print(model_args.model_name_or_path + '/' + data_args.use_adapter)
        #return
        try:
            model.load_adapter(model_args.model_name_or_path + '/' + data_args.use_adapter)
        except Exception as e:
            print('There has no a pretrained adapter model!')
            pass'''
        

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    custom_dataset_name = data_args.custom_dataset_name if data_args.custom_dataset_name is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    #metric_rouge = evaluate.load("rouge")
    #metric_bleu = evaluate.load("bleu")
    #metric_sacrebleu = evaluate.load("sacrebleu")
    #metric_meteor = evaluate.load("meteor")

    def postprocess_text(preds, labels):
        
        # remove spaces both head and tail
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence + NLTK tokenization
        #preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        #labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        
        # prefer not to use any tokenization for all metrics, except PARENT
        preds = [' '.join(pred.split()) for pred in preds]
        labels = [' '.join(label.split()) for label in labels]

        return preds, labels


    def normalize_parent(inputs, labels, preds, source_prefix, custom_dataset_name = 'wida2wl'):
        """
            custom_dataset_name: name of custom dataset
        """
    
        # prefix
        table_list, label_list, pred_list = [], [], []
        
        for inp, label, pred in zip(inputs, labels, preds):
            table = []
            
            # treat qualifiers as (key, value) pairs
            # example: "label : August 18 | part of : August > series ordinal : 18 ."
            inp = inp.replace(source_prefix, '').strip() # take out prefix   
            inp = inp.split('[SEP]')[0].strip() # get first part before [SEP]

            triples = re.split('\| |>', inp) 
            for item in triples:
                #print('triple: ', item)
                
                temp_list = [x.strip() for x in item.split(':')]
                #if (temp_list[0] not in ignore_relations): continue           
                key = temp_list[0].strip().replace(' ', '_').lower()
                
                value = ''
                try:
                    value = temp_list[1]
                    doc = nlp(value)
                    value = [token.text.lower() for token in doc] 
                except:
                    pass
                
                if (value == ''): continue
                table.append([key, value])
                
            if (len(table) == 0): continue
            table_list.append(table)
                
            doc = nlp(label)
            label = ' '.join(token.text for token in doc if token.text.strip() != '').lower()
            label = label.replace('(', '-lrb-').replace(')', '-rrb-')
            label_list.append(label.split())
            
            doc = nlp(pred)
            pred = ' '.join(token.text for token in doc if token.text.strip() != '').lower()
            pred = pred.replace('(', '-lrb-').replace(')', '-rrb-')
            pred_list.append(pred.split())
            
        return table_list, label_list, pred_list
            

    def extract_values(texts, source_prefix = '', custom_dataset_name = 'wida2wl'):

        """
            # label : August 18 | part of : August > series ordinal : 18 .
        """
        
        #print('custom_dataset_name: ', custom_dataset_name)

        value_list = []

        if (custom_dataset_name == 'wida2wl' or custom_dataset_name == 'e2e_nlg'):
            for text in texts:
                text = text.replace(source_prefix, '')
                text = text.split('[SEP]')[0].strip()

                #print('text: ', text)
                values = []        
                triples = re.split('\| |>', text)
                for item in triples:
                    try:
                        temp_list = [x.strip() for x in item.split(':')]
                        if (custom_dataset_name == 'wida2wl'):
                            if (temp_list[0] in ignore_relations): continue

                        value = temp_list[1] # get value

                        if (custom_dataset_name == 'wida2wl'):
                            try:
                                datetime = convert_datetime(value)
                                if (datetime != ''): value = str(datetime.year) # for easier matching
                            except: pass
                            
                        values.append(value)
                    except: pass
                # values = list(set(values)), values.sort(key = len, reverse = True)
                value_list.append(values)
                #print('values: ', values)
                

        if (custom_dataset_name == 'dart'):
            for text in texts:

                text = text.replace(source_prefix, '')
                values = []        
                triples = re.split('\|', text)
                for item in triples:
                    try:
                        temp_list = [x.strip() for x in item.split(':')]
                        value = temp_list[2] # get value
                        values.append(value)
                    except: pass
                value_list.append(values)

        return value_list


    def compute_metrics_detail(inputs, input_values, preds, labels, source_prefix = '', result_dict = {}):

        """
            data2text
        """

        can_tar_dict, can_source_dict = {}, {}

        parent_inputs, parent_labels, parent_preds = normalize_parent(inputs, labels, preds, source_prefix, \
                                                                      custom_dataset_name = custom_dataset_name)
        precision, recall, f1 = parent(parent_preds, parent_labels, parent_inputs, avg_results=True, n_jobs=32)
        result_dict['parent_f1'] = f1

        score = compute_repetition_batch(preds)
        result_dict['rep'] = score['rep']
   
        score = compute_spm_batch(preds, input_values)
        result_dict['spm'] = score['spm']

        score = compute_bleu_batch(preds, labels)
        can_tar_dict['bleu'] = score['bleu']
        score = compute_bleu_batch(preds, inputs)
        can_source_dict['bleu'] = score['bleu']

        score = compute_meteor_batch(preds, labels)
        can_tar_dict['meteor'] = score['meteor']
        score = compute_meteor_batch(preds, inputs)
        can_source_dict['meteor'] = score['meteor']
 
        score = compute_rouge_batch(preds, labels)
        for k, v in score.items(): can_tar_dict[k] = v
        score = compute_rouge_batch(preds, inputs)
        for k, v in score.items(): can_source_dict[k] = v

        score = compute_bertscore_batch(preds, labels)
        for k, v in score.items(): can_tar_dict[k] = v
        score = compute_bertscore_batch(preds, inputs)
        for k, v in score.items(): can_source_dict[k] = v
    
        # fuse metric
        scores = [result_dict['spm'], can_tar_dict['rouge1_precision']]
        result_dict['fused_metric'] = fuse_score(scores)

        #result_dict['prediction_vs_target'] = can_tar_dict
        #result_dict['prediction_vs_source'] = can_source_dict

        for k, v in can_tar_dict.items():
            result_dict['pred_vs_target_' + k] = v

        for k, v in can_source_dict.items():
            result_dict['pred_vs_source_' + k] = v

        return result_dict

    def compute_metrics(eval_preds):
        
        inputs = eval_preds.inputs
        labels = eval_preds.label_ids
        preds = eval_preds.predictions

        if isinstance(preds, tuple):
            preds = preds[0]
        
        if data_args.ignore_pad_token_for_loss:
            # replace -100 in the labels as we can't decode them
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
            
        #print('inputs: ', inputs[0])

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens = True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens = True)
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens = True)

        print('decoded_inputs: ', decoded_inputs[0])

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        input_values = extract_values(decoded_inputs, source_prefix = prefix, custom_dataset_name = custom_dataset_name)
        print('input_values: ', input_values[0])
        
        print(decoded_inputs[0], decoded_labels[0], decoded_preds[0])

        result_dict = {}
        result_dict = compute_metrics_detail(decoded_inputs, input_values, decoded_preds, decoded_labels, source_prefix = prefix, \
                                             result_dict = result_dict)
        
        print('result_dict: ', result_dict)
        '''
        result = {}
        
        result = compute_rouge_batch(decoded_preds, decoded_labels)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        
        parent_inputs, parent_labels, parent_preds = normalize_parent(decoded_inputs, decoded_labels, decoded_preds, prefix)  
        precision, recall, f1 = parent(parent_preds, parent_labels, parent_inputs, avg_results=True, n_jobs=32)
        
        result['parent_precision'] = precision*100
        result['parent_recall'] = recall*100
        result['parent_f1'] = f1*100
        print(precision, recall, f1)
        
        bleu = compute_bleu_batch(predictions, references)
        result['bleu'] = bleu['bleu']*100
        
        sacrebleu = metric_sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        result['sacrebleu'] = sacrebleu['score']
        
        meteor = metric_meteor.compute(predictions=decoded_preds, references=decoded_labels)
        meteor = compute_meteor_batch(decoded_preds, decoded_labels)
        result['meteor'] = meteor['meteor']*100
        '''
        
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result_dict["gen_len"] = np.mean(prediction_lens)
        return result_dict

    # Early stopping
    if data_args.patience and data_args.patience > 0:
        training_args.load_best_model_at_end = True

    # Setup training arguments
    training_args.num_train_epochs = data_args.epochs
    training_args.evaluation_strategy = 'epoch'
    training_args.save_total_limit = 1
    training_args.include_inputs_for_metrics = True # Transformers >= v4.2.0
    training_args.metric_for_best_model = 'parent_f1'
    training_args.load_best_model_at_end = True
    
    training_args.greater_is_better = True
    training_args.generation_max_length = data_args.val_max_target_length # 256 by default
    training_args.generation_min_length = data_args.val_min_target_length # 4 by default
    training_args.save_strategy = 'epoch'
    #training_args.output_dir = output_dir
    #training_args.learning_rate = 5e-5
    
    
    fp16_value = False
    if (torch.cuda.is_available() == True and 't5' not in model_args.model_name_or_path): fp16_value = True
    training_args.fp16 = fp16_value

  
    # Setup adapters
    if (data_args.use_adapter != ''): 
    
    
        if (data_args.use_adapter == 'bottleneck'):
            from transformers.adapters import AdapterConfig
            aconfig = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=2, non_linearity="relu")
            model.add_adapter(data_args.use_adapter, config=aconfig)
            
        if (data_args.use_adapter == 'prefix'):
            from transformers.adapters import PrefixTuningConfig
            aconfig = PrefixTuningConfig(flat=False, prefix_length=30) # False
            model.add_adapter(data_args.use_adapter, config=aconfig)
            #model.eject_prefix_tuning(data_args.use_adapter) # reduce parameters + size
        
        if (data_args.use_adapter == 'lang'):
            from transformers.adapters import PfeifferInvConfig
            aconfig = PfeifferInvConfig()
            model.add_adapter(data_args.use_adapter, config=aconfig) 
        
        if (data_args.use_adapter == 'lora'): 
            from transformers.adapters import LoRAConfig
            aconfig = LoRAConfig(r=8, alpha=16)
            model.add_adapter(data_args.use_adapter, config=aconfig)
            #model.merge_adapter(data_args.use_adapter)
            #model.reset_adapter(data_args.use_adapter)
        
        if (data_args.use_adapter == 'mam'):
            from transformers.adapters import MAMConfig
            aconfig = MAMConfig()
            model.add_adapter(data_args.use_adapter, config=aconfig)
            '''
            from transformers.adapters import ConfigUnion, ParallelConfig, PrefixTuningConfig
            config = ConfigUnion(
                PrefixTuningConfig(bottleneck_size=800),
                ParallelConfig(),
            )
            model.add_adapter("mam_adapter", config=config)
            '''
        
        if (data_args.use_adapter == 'unipelt'): # not yet
            
            from transformers.adapters import UniPELTConfig
            aconfig = UniPELTConfig()
            model.add_adapter(data_args.use_adapter, config=aconfig)
            '''
            from transformers.adapters import ConfigUnion, LoRAConfig, PrefixTuningConfig, PfeifferConfig
            config = ConfigUnion(
                LoRAConfig(r=8, use_gating=True),
                PrefixTuningConfig(prefix_length=10, use_gating=True),
                PfeifferConfig(reduction_factor=16, use_gating=True),
            )
            model.add_adapter("unipelt", config=config)
            '''
        
        if (data_args.use_adapter == 'union'): 
        
            from transformers.adapters import AdapterConfig, ParallelConfig, ConfigUnion

            config = ConfigUnion(
                ParallelConfig(mh_adapter=True, output_adapter=False, reduction_factor=16, non_linearity="relu"),
                ParallelConfig(mh_adapter=False, output_adapter=True, reduction_factor=2, non_linearity="relu"),
            )
            model.add_adapter(data_args.use_adapter, config=config)
        
        if (data_args.use_adapter == 'union2'):
            from transformers.adapters import AdapterConfig, ConfigUnion, ParallelConfig, PrefixTuningConfig
            config = ConfigUnion(
                AdapterConfig(mh_adapter=True, output_adapter=False, reduction_factor=16, non_linearity="relu"),
                #AdapterConfig(mh_adapter=False, output_adapter=True, reduction_factor=2, non_linearity="relu"),
                
                #PrefixTuningConfig(bottleneck_size=800),
                ParallelConfig()
            )
            model.add_adapter(data_args.use_adapter, config=config)
        
        
        if (data_args.use_adapter == 'fuse'):
        
            from transformers.adapters import ParallelConfig
            aconfig = ParallelConfig()
            model.add_adapter('adapter1', config=aconfig)
            
            '''from transformers.adapters import CompacterConfig
            aconfig = CompacterConfig()
            model.add_adapter('adapter2', config=aconfig)'''
            
            from transformers.adapters import ParallelConfig
            aconfig = ParallelConfig(reduction_factor=8)
            model.add_adapter('adapter2', config=aconfig)

            from transformers.adapters.composition import Fuse
            adapter_setup = Fuse('adapter1', 'adapter2')
            model.set_active_adapters(adapter_setup)
            
            model.add_adapter_fusion(adapter_setup)
            model.add_seq2seq_lm_head('adapter1,adapter2', True)
            model.train_adapter_fusion(adapter_setup)

        if (data_args.use_adapter == 'parallel'):
        
            # under construction
            pass
            '''from transformers.adapters import AdapterConfig
            aconfig = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=2, non_linearity="relu")
            model.add_adapter('bottleneck', config=aconfig)
            
            from transformers.adapters import CompacterConfig
            aconfig = CompacterConfig()
            model.add_adapter('dummy', config=aconfig)

            from transformers.adapters.composition import Parallel
            adapter_setup = Parallel('bottleneck', 'dummy')
            
            model.set_active_adapters(adapter_setup)
            
            model.add_adapter_fusion(adapter_setup)
            model.add_seq2seq_lm_head('bottleneck,dummy', True)
            model.train_adapter_fusion(adapter_setup)'''
       
        if (data_args.use_adapter not in ['fuse', 'parallel']):
            model.set_active_adapters(data_args.use_adapter)
            model.add_seq2seq_lm_head(data_args.use_adapter, True)
            model.train_adapter(data_args.use_adapter, True)
        
        setup_adapter_training(model, adapter_args, data_args.dataset_name or "summarization")
    
    
    # Initialize predict only   
    if training_args.do_predict and not training_args.do_train and not training_args.do_eval:
        if (data_args.use_adapter == 'fuse'):
            model.load_adapter_fusion(training_args.output_dir.strip('/') + '/adapter1,adapter2/')
            
        elif (data_args.use_adapter != ''):
            model.load_adapter(training_args.output_dir + '/' + data_args.use_adapter)
            model.set_active_adapters(data_args.use_adapter)
    
    # Initialize our Trainer  
    trainer_class = Seq2SeqAdapterTrainer if data_args.use_adapter != '' else Seq2SeqTrainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )
    
    if data_args.patience and data_args.patience > 0:
        callback = EarlyStoppingCallback(early_stopping_patience=data_args.patience)
        trainer.add_callback(callback)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    min_length = (
        training_args.generation_min_length
        if training_args.generation_min_length is not None
        else data_args.val_min_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
            
        #print(dir(trainer), trainer)
        #return
        
        #model.load_adapter("adapter_poem") # --adapter_path
        #model.set_active_adapters("poem") # --use_adapter

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, min_length = min_length, \
            num_beams=num_beams
        )
        
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()
        
if __name__ == "__main__":
    main()
