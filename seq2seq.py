import gc
#torch.cuda.empty_cache()
#gc.collect()

import torch
import os
from torch import nn
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
device_cpu = torch.device('cpu')

'''torch.cuda.set_per_process_memory_fraction(0.5, 0)
torch.cuda.empty_cache()
total_memory = torch.cuda.get_device_properties(0).total_memory'''

import time

import re
import sys
import math
import random
import numpy as np
import statistics

import shutil

import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from transformers import BertModel, BertTokenizer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)

#import spacy
#nlp = spacy.load('en_core_web_md')
#nlp.add_pipe('sentencizer', before='parser') # for spaCy 3.x.x

import nltk
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)
        
from nltk.tokenize import wordpunct_tokenize, word_tokenize

from parent import parent # install https://github.com/KaijuML/parent

#from transformers import Trainer
from metrics import *
from file_io import *

vocab_dict = {} # use for another works
#try: vocab_dict = load_list_from_json_file('dataset/vocab.json')
#except: pass

#ignore_relations = ['instance of', 'sex or gender', 'country of citizenship']  # familyFriendly (dart and e2e_nlg)
ignore_relations  = []

class CustomTrainer(Seq2SeqTrainer):
        
    def compute_loss(self, model, inputs, return_outputs=False):
        
        """
            loss: Optional[torch.FloatTensor] = None
            logits: torch.FloatTensor = None
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
            decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
            decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
            cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
            encoder_last_hidden_state: Optional[torch.FloatTensor] = None
            encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
            encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
        """
        
        if self.label_smoother is not None and "labels" in inputs: labels = inputs.pop("labels")
        else: labels = None
        outputs = model(**inputs)
        
        # save past state if it exists
        if self.args.past_index >= 0: self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
                
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        base_setting(args)
        if (use_fuse_loss == True): # this loss does not help the model better
            
            source_ids = inputs.get('input_ids')
            source_ids =[[tokenizer.pad_token_id if token == -100 else token for token in l]
                      for l in source_ids]
            sources = tokenizer.batch_decode(source_ids, skip_special_tokens = True)

            attention_mask = inputs.get('attention_mask')
            pred_ids = []
            with torch.no_grad():
                pred_ids = self.model.generate(inputs.get('input_ids'), attention_mask = attention_mask, \
                                               max_length = args.decoder_max_length, min_length = args.decoder_min_length)
            pred_ids = [[tokenizer.pad_token_id if token == -100 else token for token in l]
                      for l in pred_ids]
            preds = tokenizer.batch_decode(pred_ids, skip_special_tokens = True)

            targets_ids = inputs.get('labels')
            targets_ids = [[tokenizer.pad_token_id if token == -100 else token for token in l]
                      for l in targets_ids]
            targets = tokenizer.batch_decode(targets_ids, skip_special_tokens = True)
        
            input_values = extract_values(sources)
        
            spm = compute_spm_batch(preds, input_values)['spm']
            #non_rep = 1 - compute_repetition_batch(preds)['rep']
            rouge1_precision = compute_rouge_batch(preds, targets)['rouge1_precision']
            fuse_loss = 1 - fuse_score([spm, rouge1_precision])

            alpha = 0.9 # trade-off between 2 losses
            loss = loss*alpha + (1-alpha)*fuse_loss
            
        return (loss, outputs) if return_outputs else loss

def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length, \
                              source_column = 'source', target_column = 'target', source_prefix = 'summarize# '):
    source, target = batch[source_column], batch[target_column]
    source = [source_prefix + item for item in source]
    
    source_tokenized = tokenizer(source, padding="max_length", truncation=True, max_length=max_source_length)
    batch = {k: v for k, v in source_tokenized.items()}
    target_tokenized = tokenizer(target, padding="max_length", truncation=True, max_length=max_target_length)

    # Ignore padding in the loss
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in l]
                       for l in target_tokenized["input_ids"]]                
    return batch

def load_data(tokenizer, batch_size, encoder_max_length, decoder_max_length,  train_file = '', val_file = '', \
              source_column = 'source', target_column = 'target', source_prefix = 'summarize# '):

    """
        batch_size, encoder_max_length, decoder_max_length,  train_file = '', val_file = '', \
              source_column = 'source', target_column = 'target'
              
        train_data, val_data = load_data(tokenizer, args.batch_size, args.encoder_max_length, \
                                         args.decoder_max_length, train_file = args.train_file, \
                                         val_file = args.var_file, source_column = args.source_column, \
                                         target_column = args.target_column)
    """
    
    train_data = datasets.load_dataset('json', data_files = train_file)
    #train_data = datasets.load_dataset('json', data_files = train_file, split='train[:5%]')
    
    val_data = datasets.load_dataset('json', data_files = val_file)
    
    train_data = train_data.map(lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, encoder_max_length, decoder_max_length, source_column, target_column, source_prefix),
            batched=True,
            batch_size=batch_size,
            remove_columns = train_data['train'].column_names
        )

    val_data = val_data.map(lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, encoder_max_length, decoder_max_length, source_column, target_column, source_prefix),
            batched=True,
            batch_size=batch_size,
            remove_columns = val_data['train'].column_names
        )

    print('train_data: ', len(train_data['train']))
    print('val_data: ', len(val_data['train']))
    return train_data, val_data

def load_data2(tokenizer, batch_size, encoder_max_length, decoder_max_length, data_list, \
               source_column = 'source', target_column = 'target', source_prefix = 'summarize# '):
    
    #train_data = datasets.Dataset.from_dict(data_dict)
    train_data = datasets.Dataset.from_pandas(pd.DataFrame(data=data_list))

    train_data = train_data.map(lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, encoder_max_length, decoder_max_length, source_column, target_column, source_prefix),
            batched=True,
            batch_size=batch_size,
            #remove_columns=['source', 'target']
            remove_columns = train_data.column_names
        )

    # add key 'train'    
    new_train_data = {}
    new_train_data['train'] = train_data
    
    print('train_data: ', len(new_train_data['train']))
    return new_train_data


def extract_values(texts, dataset_name = 'wida2wl'):

    """
        # label : August 18 | part of : August > series ordinal : 18
    """

    value_list = []

    if (dataset_name == 'wida2wl' or dataset_name == 'e2e_nlg'):
        for text in texts:
            values = []        
            triples = re.split('\| |>', text)
            for item in triples:
                try:
                    temp_list = [x.strip() for x in item.split(' : ')]
                    
                    #if (dataset_name == 'wida2wl'):
                    if (temp_list[0] in ignore_relations): continue

                    value = temp_list[1] # get value

                    if (dataset_name == 'wida2wl'):
                        try:
                            datetime = convert_datetime(value)
                            if (datetime != ''): value = str(datetime.year) # for easier matching
                        except: pass
                        
                    values.append(value)
                except: pass
            # values = list(set(values)), values.sort(key = len, reverse = True)
            value_list.append(values)

    if (dataset_name == 'dart'):
        for text in texts:
            values = []        
            triples = re.split('\|', text)
            for item in triples:
                try:
                    temp_list = [x.strip() for x in item.split(' : ')]
                    if (temp_list[1] in ignore_relations): continue
                    
                    value = temp_list[0] # get subject
                    values.append(value)
                    
                    value = temp_list[2] # get object
                    values.append(value)
                except: pass
            value_list.append(values)
            #print(values)
    
    return value_list

def normalize_parent(inputs, labels, preds, dataset_name = 'wida2wl'):

    """
        wida2wl: "label : August 18 | part of : August > series ordinal : 18 ."
    """

    table_list, label_list, pred_list = [], [], []

    for inp, label, pred in zip(inputs, labels, preds):
        table = []
        triples = re.split('\| |>', inp)
        
        for item in triples:
            temp_list = [x.strip() for x in item.split(':')]

            #if (dataset_name == 'wida2wl'):
                #if (temp_list[0] in ignore_relations): continue

            key, value = '', ''
            
            try:
                if (dataset_name == 'wida2wl' or dataset_name == 'e2e_nlg'):
                    key = temp_list[0].replace(' ', '_').lower()
                    value = temp_list[1].strip()
                if (dataset_name == 'dart'):
                    key = temp_list[0].replace(' ', '_').lower() + '_' + temp_list[1].replace(' ', '_').lower()
                    value = temp_list[2].strip()
                    #print(key, '***', value)
            except:
                pass
            
            if (value == '' or key == ''): continue

            #doc = nlp(value) # spaCy
            #value = [token.text.lower() for token in doc]
            
            doc = word_tokenize(value) # observations from https://github.com/KaijuML/parent/tree/master/data
            value = [token.lower() for token in doc]
            table.append([key, value])
        
        if (len(table) == 0): continue
        table_list.append(table)
                
        doc = word_tokenize(label) # observations from https://github.com/KaijuML/parent/tree/master/data
        label = ' '.join(token for token in doc if token.strip() != '').lower()
        label = label.replace('(', '-lrb-').replace(')', '-rrb-')
        label_list.append(label.split())
            
        doc = word_tokenize(pred) # observations from https://github.com/KaijuML/parent/tree/master/data
        pred = ' '.join(token for token in doc if token.strip() != '').lower()
        pred = pred.replace('(', '-lrb-').replace(')', '-rrb-')
        pred_list.append(pred.split())
           
    return table_list, label_list, pred_list


def compute_metrics(pred):

    """
        data to text
    """

    input_ids = pred.inputs # Transformers >= v4.2.0
    input_ids[input_ids == -100] = tokenizer.pad_token_id
    inputs = tokenizer.batch_decode(input_ids, skip_special_tokens = True)
    #inputs = [inp.replace('<pad>', '').replace('</s>', '') for inp in inputs]
    inputs = [inp.replace(source_prefix, '') for inp in inputs]
    #inputs = [inp.split('[SEP]')[0].strip() for inp in inputs] # for inputs in self-train
    
    '''
    for inp, ids in zip(inputs, input_ids):
        print(inp, ids)
        print('----------------')
    return'''
    
    pred_ids = pred.predictions
    pred_ids[pred_ids == -100] = tokenizer.pad_token_id
    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens = True)
    #preds = [pred.replace('<pad>', '').replace('</s>', '') for pred in preds]

    labels_ids = pred.label_ids
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    labels = tokenizer.batch_decode(labels_ids, skip_special_tokens = True)
    #labels = [label.replace('<pad>', '').replace('</s>', '')for label in labels]

    # tokenize by NLTK
    if (dataset_name == 'dart'):
        for i in range(0, len(labels)):
            doc = word_tokenize(inputs[i])
            inputs[i] = ' '.join(token for token in doc if token.strip() != '')
            doc = word_tokenize(preds[i])
            preds[i] = ' '.join(token for token in doc if token.strip() != '')
            doc = word_tokenize(labels[i])
            labels[i] = ' '.join(token for token in doc if token.strip() != '')
                
    # extract values
    input_values = extract_values(inputs, dataset_name = dataset_name) # dataset_name is a global variable
    
    # print first item
    print('inputs: ', len(inputs), inputs[0]) # source
    print('input_values: ', len(input_values), input_values[0]) # source values
    print('preds: ', len(preds), preds[0]) # prediction
    print('labels: ', len(labels), labels[0]) # target

    # dataset_name is a global variable
    result_dict = {}
    result_dict = compute_metrics_detail(inputs, input_values, preds, labels, result_dict, dataset_name = dataset_name)
    return result_dict


def compute_metrics2(pred):

    """
        text to data
    """

    input_ids = pred.inputs # Transformers >= v4.2.0
    input_ids[input_ids == -100] = tokenizer.pad_token_id
    inputs = tokenizer.batch_decode(input_ids, skip_special_tokens = True)
    #inputs = [inp.split(' [SEP] ')[0] for inp in inputs] # for inputs in self-train
    
    pred_ids = pred.predictions
    pred_ids[pred_ids == -100] = tokenizer.pad_token_id
    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens = True)

    labels_ids = pred.label_ids
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    labels = tokenizer.batch_decode(labels_ids, skip_special_tokens = True)

    # tokenize by NLTK
    if (dataset_name == 'dart'):
        for i in range(0, len(labels)):
            doc = word_tokenize(inputs[i])
            inputs[i] = ' '.join(token for token in doc if token.strip() != '')
            doc = word_tokenize(preds[i])
            preds[i] = ' '.join(token for token in doc if token.strip() != '')
            doc = word_tokenize(labels[i])
            labels[i] = ' '.join(token for token in doc if token.strip() != '')

    # extract values
    preds_values = extract_values(preds, dataset_name = dataset_name) # dataset_name is a global variable
    
    # print first item
    print('inputs: ', len(inputs), inputs[0]) # source
    print('preds: ', len(preds), preds[0]) # prediction
    print('preds_values: ', len(preds_values), preds_values[0]) # pred values
    print('labels: ', len(labels), labels[0]) # target

    result_dict = {}
    result_dict = compute_metrics_detail2(inputs, preds, preds_values, labels, result_dict)
         
    return result_dict


def train(model_name, model, tokenizer, train_data, val_data, num_train_epochs = 3, batch_size = 4, output_dir = 'output/', \
          generation_max_len = 256, train_type = 'd2t', dataset_name = ''):
    
    save_steps = len(train_data['train'])//batch_size
    print('save_steps: ', save_steps)
    #save_steps = len(train_data['train'])
    warmup_steps = save_steps//10 # 10% warmup
    
    output_dir = output_dir + '/' + dataset_name
    output_dir = output_dir.replace('//', '/')

    fp16_value = False 
    if (torch.cuda.is_available() == True and 't5' not in model_name): fp16_value = True
    #if (torch.cuda.is_available() == True): fp16_value = True

    training_args = Seq2SeqTrainingArguments(
        gradient_accumulation_steps = 1,
        #gradient_checkpointing=True,
        include_inputs_for_metrics = True,  # Transformers >= v4.2.0
        #weight_decay=0.1,
        #label_smoothing_factor=0.1,
        #logging_dir="logs",
        #learning_rate=3e-05,
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        generation_max_length=generation_max_len,
        #evaluate_during_training=True,
        evaluation_strategy='epoch',
        do_train=True,
        do_eval=True,
        logging_steps = save_steps,  
        #save_steps = True, 
        #eval_steps = save_steps,  
        warmup_steps = warmup_steps,  
        #max_steps = 16,
        overwrite_output_dir = True,
        save_total_limit = 2,
        save_strategy = 'epoch',
        num_train_epochs = num_train_epochs,
        fp16 = fp16_value,  # cuda only
        metric_for_best_model = eval_metric if train_type == 'd2t' else 'eval_osf_f1', 
        load_best_model_at_end = True, # will ignore save steps, save after each evaluation
        logging_dir=f"{output_dir}/logs",
        logging_strategy="epoch", 
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model = model)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data['train'],
        eval_dataset=val_data['train'],
        tokenizer=tokenizer,
        compute_metrics = compute_metrics if train_type == 'd2t' else compute_metrics2
        #callbacks=[CustomCallback]
    )
   
    trainer.train()
    #trainer.evaluate()
    
    #trainer.save_state()
    #trainer.save_model(output_dir)

    # see inside the train object
    #import inspect
    #from pprint import pprint
    #pprint(inspect.getmembers(trainer))
    #print(dir(trainer))

    return trainer.state


def generate_single(model, tokenizer, text, max_length = 256, min_length = 4, num_beams = 5, decoding_type = 'beam', \
                    force_text = ''):

    inputs = tokenizer(text, padding = "max_length", truncation = True, max_length = max_length, return_tensors = "pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    if (num_beams < 2): num_beams = 2

    force_words_ids = []
    if (force_text != ''):
        #force_words_ids = tokenizer.encode(force_text, padding = "max_length", max_length = max_length)
        force_words_ids = tokenizer(force_text, add_special_tokens=False).input_ids
        force_words_ids = [ids for ids in force_words_ids if ids != -100]
    
    
    if (decoding_type == 'multinomial'):
        with torch.no_grad():
        
            if (len(bad_words) == 0):
                outputs = model.generate(input_ids, attention_mask = attention_mask, \
                                     max_length = max_length, min_length = min_length, num_beams = 1, \
                                     do_sample = True, return_dict_in_generate=True, output_scores=True)  
            else:
                outputs = model.generate(input_ids, bad_words_ids = [bad_words], attention_mask = attention_mask, \
                                     max_length = max_length, min_length = min_length, num_beams = 1, \
                                     do_sample = True, return_dict_in_generate=True, output_scores=True)  
    elif (decoding_type == 'beam'):
        with torch.no_grad():
            if (len(force_words_ids) == 0):
                if (len(bad_words) == 0):
                    outputs = model.generate(input_ids, attention_mask = attention_mask, \
                                         max_length = max_length, min_length = min_length, num_beams = num_beams, \
                                         do_sample = False, return_dict_in_generate=True, output_scores=True)
                else:
                    outputs = model.generate(input_ids, bad_words_ids = [bad_words], attention_mask = attention_mask, \
                                         max_length = max_length, min_length = min_length, num_beams = num_beams, \
                                         do_sample = False, return_dict_in_generate=True, output_scores=True)
            else:
                if (len(bad_words) == 0):
                    outputs = model.generate(input_ids, force_words_ids = [force_words_ids], \
                                         attention_mask = attention_mask, \
                                         max_length = max_length, min_length = min_length, num_beams = num_beams, \
                                         do_sample = False, return_dict_in_generate=True, output_scores=True)
                else:
                    outputs = model.generate(input_ids, force_words_ids = [force_words_ids], bad_words_ids = [bad_words], \
                                         attention_mask = attention_mask, \
                                         max_length = max_length, min_length = min_length, num_beams = num_beams, \
                                         do_sample = False, return_dict_in_generate=True, output_scores=True)

    elif (decoding_type == 'beam_multinomial'):
        with torch.no_grad():
            if (len(force_words_ids) == 0):
                if (len(bad_words) == 0):
                    outputs = model.generate(input_ids, attention_mask = attention_mask, \
                                         max_length = max_length, min_length = min_length, num_beams = num_beams, \
                                         do_sample = True, return_dict_in_generate=True, output_scores=True)
                else:
                    outputs = model.generate(input_ids, bad_words_ids = [bad_words], attention_mask = attention_mask, \
                                         max_length = max_length, min_length = min_length, num_beams = num_beams, \
                                         do_sample = True, return_dict_in_generate=True, output_scores=True)
            else:
                if (len(bad_words) == 0):
                    outputs = model.generate(input_ids, force_words_ids = [force_words_ids], \
                                         attention_mask = attention_mask, \
                                         max_length = max_length, min_length = min_length, num_beams = num_beams, \
                                         do_sample = True, return_dict_in_generate=True, output_scores=True)
                else:
                    outputs = model.generate(input_ids, force_words_ids = [force_words_ids], bad_words_ids = [bad_words], \
                                             attention_mask = attention_mask, max_length = max_length, min_length = min_length, \
                                             num_beams = num_beams, do_sample = True, return_dict_in_generate=True, \
                                             output_scores=True)
    else:
        # greedy
        # suppress_tokens = bad_words
        with torch.no_grad():
            if (len(bad_words) == 0):
                outputs = model.generate(input_ids, attention_mask = attention_mask, \
                                        max_length = max_length, min_length = min_length, num_beams = 1, do_sample = False, \
                                        return_dict_in_generate=True, output_scores=True)
            else:
                outputs = model.generate(input_ids, bad_words_ids = [bad_words], attention_mask = attention_mask, \
                                        max_length = max_length, min_length = min_length, num_beams = 1, do_sample = False, \
                                        return_dict_in_generate=True, output_scores=True)

    # calculate prob  
    '''print(len(outputs.scores))
    sum_score = 0
    for score in outputs.scores:
        prob_score = torch.nn.functional.softmax(score, dim = 1)
        sum_score += torch.log(torch.sum(prob_score))'''

    pred_text = ''
    try:
        pred_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens = True)
        pred_text = pred_text[0]
    except: pass
    
    return pred_text

def generate_batch(model, tokenizer, texts, force_texts = [], max_length = 256, min_length = 4, num_beams = 4, \
                   decoding_type = 'beam'):

    pred_texts = []
    if (len(force_texts) == 0):

        inputs = tokenizer(texts, padding = "max_length", truncation = True, max_length = max_length, \
                           return_tensors = 'pt').to(device)
        output = {}
        if (decoding_type == 'multinomial'):
            with torch.no_grad():
                if (len(bad_words) == 0):
                    outputs = model.generate(**inputs, max_length = max_length, min_length = min_length, \
                                             num_beams = 1, do_sample = True, \
                                             return_dict_in_generate = True, output_scores = True)
                else:
                    outputs = model.generate(**inputs, bad_words_ids = [bad_words], max_length = max_length, \
                                             min_length = min_length, num_beams = 1, do_sample = True, \
                                             return_dict_in_generate = True, output_scores = True)
        elif (decoding_type == 'beam'):
            with torch.no_grad():
                if (len(bad_words) == 0):
                    outputs = model.generate(**inputs, max_length = max_length, min_length = min_length, num_beams = num_beams, \
                                             do_sample = False, return_dict_in_generate = True, output_scores = True)
                else:
                    outputs = model.generate(**inputs, bad_words_ids = [bad_words], max_length = max_length, \
                                             min_length = min_length, num_beams = num_beams, do_sample = False, \
                                             return_dict_in_generate = True, output_scores = True)
        elif (decoding_type == 'beam_multinomial'):
            with torch.no_grad():
                if (len(bad_words) == 0):
                    outputs = model.generate(**inputs, max_length = max_length, min_length = min_length, num_beams = num_beams, \
                                 do_sample = True, return_dict_in_generate = True, output_scores = True)
                else:
                    outputs = model.generate(**inputs, bad_words_ids = [bad_words], max_length = max_length, \
                                             min_length = min_length, num_beams = num_beams, do_sample = True, \
                                             return_dict_in_generate = True, output_scores = True)
        else:
            # greedy
            with torch.no_grad():
                if (len(bad_words) == 0):
                    #print('bad_words: ', [bad_words])
                    outputs = model.generate(**inputs, max_length = max_length, min_length = min_length, \
                                         num_beams = 1, do_sample = False, return_dict_in_generate = True, output_scores = True)  
                else:
                    outputs = model.generate(**inputs, bad_words_ids = [bad_words], max_length = max_length, \
                                             min_length = min_length, num_beams = 1, do_sample = False, \
                                             return_dict_in_generate = True, output_scores = True) 
        pred_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens = True)
    else:

        for text, force_text in zip(texts, force_texts):
            pred = generate_single(model, tokenizer, text, max_length = max_length, min_length = min_length, \
                                   num_beams = num_beams, decoding_type = 'beam', force_text = force_text)
            pred_texts.append(pred)
        
    return pred_texts


def get_force_words(subset, n_words = 1): # not use...

    """
        item: item dictionary
        n_words: number of force words
    """

    force_texts = []
    values = [item['source_list'] for item in subset]
    for value in values:
        item_list = [] 
        for x in value:
            if (x[0] in ignore_relations): continue
            item_list.append(x[1])
            if (len(x) == 4): item_list.append(x[3]) # add qualifiers
                        
        item_list = list(set(item_list)) # remove repetitions

        # check word frequency
        words = ' '.join(w for w in item_list).split()
        rare_words = []
        for word in words:
            try:
                datetime = convert_datetime(word)
                if (datetime != ''): word = str(datetime.year)
            except Exception as e:
                print('Error -- get_force_words: ', e)
                pass
            #try:
            #if (vocab_dict[word] < 5): rare_words.append(word)
            #except: 
            rare_words.append(word)

        rare_words = list(set(rare_words))
        print('rare_words: ', rare_words)
                    
        force_texts.append(' '.join(w for w in rare_words[:n_words]))

    return force_texts


def generate_single_batch(subset, texts, idx, total_batch, model_name, model, tokenizer, use_force_words = False, \
                            decoding_type = 'greedy', num_beam = 4, max_len = 256, min_len = 4):
    
    force_texts = []   

    #print('idx: ', idx)
    sys.stdout.write('Infer batch: %d/%d \t Model: %s \r' % (idx, total_batch, model_name))
    sys.stdout.flush()
    
    preds = []
    if (use_force_words == True): force_texts = get_force_words(subset)
                
    if (decoding_type == 'beam'):
        sub_list = []
        for j in range(2, num_beam + 1):
            preds = generate_batch(model, tokenizer, texts, force_texts = force_texts, max_length = max_len, \
                                   min_length = min_len, num_beams = j, decoding_type = decoding_type)
            sub_list.append(preds)
        sub_list = list(map(list, zip(*sub_list)))
        sub_list = [list(set(x)) for x in sub_list]
        preds = sub_list
    elif (decoding_type == 'beam_multinomial'):
        sub_list = []
        for j in range(2, num_beam + 1):
            preds = generate_batch(model, tokenizer, texts, force_texts = force_texts, max_length = max_len, \
                                   min_length = min_len, num_beams = j, decoding_type = decoding_type)
            sub_list.append(preds)
        sub_list = list(map(list, zip(*sub_list)))
        sub_list = [list(set(x)) for x in sub_list]
        preds = sub_list
    elif (decoding_type == 'multinomial'):
        preds = generate_batch(model, tokenizer, texts, max_length = max_len, min_length = min_len, \
                                   num_beams = 1, decoding_type = decoding_type)
        preds = [[x] for x in preds]
        
    else: # greedy
        preds = generate_batch(model, tokenizer, texts, max_length = max_len, min_length = min_len, \
                                   num_beams = 1, decoding_type = decoding_type)
        preds = [[x] for x in preds]
        
    torch.cuda.empty_cache()
    gc.collect() 
    
    return {'index':idx, 'value':preds}


def generate_dataset(dataset, model_name, model, tokenizer, use_force_words = False, input_file = 'dataset/test_random.json', \
                     decoding_type = 'greedy', num_beam = 4, batch_size = 4, max_len = 256, min_len = 4, \
                     source_column = 'source', source_prefix = '', dataset_name = 'wida2wl', infer_max_workers = 4):

    
    #model = model.to_bettertransformer() # speed up inference, https://huggingface.co/docs/transformers/perf_infer_gpu_one
    if (len(dataset) == 0): # load dataset if not given
        dataset = read_list_from_jsonl_file(input_file)

    for item in dataset:
        #item['source'] = source_prefix + item['source']
        item[source_column] = source_prefix + item[source_column]
    
    pred_list = []
    subset_list = []
    texts_list = []
    
    if (infer_multi_thread == False):
        for i in range(0, len(dataset), batch_size):

            n_batch = 0
            if (len(dataset)%batch_size != 0): n_batch = len(dataset)//batch_size + 1
            else: n_batch = len(dataset)//batch_size
            
            sys.stdout.write('Infer batch: %d/%d \t Model: %s \r' % (i//batch_size + 1, n_batch, model_name))
            sys.stdout.flush()
            
            subset = dataset[i:i + batch_size]
            texts = [item[source_column] for item in subset]

            force_texts = []    
            if (use_force_words == True): force_texts = get_force_words(subset)
                
            if (decoding_type == 'beam'):
                sub_list = []
                for j in range(2, num_beam + 1):
                    preds = generate_batch(model, tokenizer, texts, force_texts = force_texts, max_length = max_len, \
                                       min_length = min_len, num_beams = j, decoding_type = decoding_type)
                    sub_list.append(preds)
                sub_list = list(map(list, zip(*sub_list)))
                sub_list = [list(set(x)) for x in sub_list]
                pred_list += sub_list
            elif (decoding_type == 'beam_multinomial'):
                sub_list = []
                for j in range(2, num_beam + 1):
                    preds = generate_batch(model, tokenizer, texts, force_texts = force_texts, max_length = max_len, \
                                       min_length = min_len, num_beams = j, decoding_type = decoding_type)
                    sub_list.append(preds)
                sub_list = list(map(list, zip(*sub_list)))
                sub_list = [list(set(x)) for x in sub_list]
                pred_list += sub_list
            elif (decoding_type == 'multinomial'):
                preds = generate_batch(model, tokenizer, texts, max_length = max_len, min_length = min_len, \
                                       num_beams = 1, decoding_type = decoding_type)
                preds = [[x] for x in preds]
                pred_list += preds
            else: # greedy
                preds = generate_batch(model, tokenizer, texts, max_length = max_len, min_length = min_len, \
                                       num_beams = 1, decoding_type = decoding_type)
                preds = [[x] for x in preds]
                pred_list += preds
        
            torch.cuda.empty_cache()
            gc.collect()
        pred_list = [[x.strip() for x in pred] for pred in pred_list]
    

    else:
    
        for i in range(0, len(dataset), batch_size):

            n_batch = 0
            if (len(dataset)%batch_size != 0): n_batch = len(dataset)//batch_size + 1
            else: n_batch = len(dataset)//batch_size
            subset = dataset[i:i + batch_size]
            texts = [item[source_column] for item in subset]
        
            subset_list.append(subset)
            texts_list.append(texts)
    
        idx_list = [i for i in range(0, len(subset_list))]
        len_list = len(subset_list)
    
        # speed up inference
        with ThreadPoolExecutor(max_workers = infer_max_workers) as executor:
            pred_list = executor.map(generate_single_batch, subset_list, texts_list, idx_list, [len_list]*len_list,  \
                            [model_name]*len_list,[model]*len_list, [tokenizer]*len_list, [use_force_words]*len_list, \
                            [decoding_type]*len_list, [num_beam]*len_list, [max_len]*len_list, [min_len]*len_list, timeout = 600)
    
        pred_list = sorted(pred_list, key=lambda p: p['index']) 
        pred_list = [pred['value'] for pred in pred_list]
        pred_list = sum(pred_list, [])  
        pred_list = [[x.strip() for x in pred] for pred in pred_list]
        
    print('pred_list: ', pred_list[0], len(pred_list))
    
    return pred_list


def compute_metrics_detail(inputs, input_values, preds, labels, result_dict = {}, dataset_name = 'wida2wl', mode = 'train'):

    """
        data2text
            "dataset_name" is a global variable
    """
    
    can_tar_dict, can_source_dict = {}, {}

    if (mode == 'train'): 
        print('Calculating SPM...')
        score = compute_spm_batch(preds, input_values)
        result_dict['spm'] = score['spm']

        '''print('Calculating PARENT...')
        parent_inputs, parent_labels, parent_preds = normalize_parent(inputs, labels, preds, dataset_name = dataset_name)
        precision, recall, f1 = parent(parent_preds, parent_labels, parent_inputs, avg_results=True, n_jobs=16)
        result_dict['parent'] = f1'''

        print('Calculating BLEU...')
        score = compute_bleu_batch(preds, labels)
        result_dict['bleu'] = score['bleu']

        print('Calculating METEOR...')
        score = compute_meteor_batch(preds, labels)
        result_dict['meteor'] = score['meteor']
    
        return result_dict

    print('Calculating SPM...')
    score = compute_spm_batch(preds, input_values)
    result_dict['spm'] = score['spm']
    
    '''print('Calculating PARENT...')
    parent_inputs, parent_labels, parent_preds = normalize_parent(inputs, labels, preds, dataset_name = dataset_name)
    precision, recall, f1 = parent(parent_preds, parent_labels, parent_inputs, avg_results=True, n_jobs=32)
    result_dict['parent'] = f1'''

    print('Calculating REP...')
    score = compute_repetition_batch(preds)
    result_dict['rep'] = score['rep']

    print('Calculating BLEU...')
    score = compute_bleu_batch(preds, labels)
    can_tar_dict['bleu'] = score['bleu']
    score = compute_bleu_batch(preds, inputs)
    can_source_dict['bleu'] = score['bleu']

    print('Calculating METEOR...')
    score = compute_meteor_batch(preds, labels)    
    can_tar_dict['meteor'] = score['meteor']
    score = compute_meteor_batch(preds, inputs)
    can_source_dict['meteor'] = score['meteor']
 
    print('Calculating ROUGE...')
    score = compute_rouge_batch(preds, labels)
    for k, v in score.items(): can_tar_dict[k] = v
    score = compute_rouge_batch(preds, inputs)
    for k, v in score.items(): can_source_dict[k] = v

    print('Calculating BERTSCORE...')
    score = compute_bertscore_batch(preds, labels)
    for k, v in score.items(): can_tar_dict[k] = v
    score = compute_bertscore_batch(preds, inputs)
    for k, v in score.items(): can_source_dict[k] = v
    
    # fused metric
    scores = [result_dict['spm'], can_tar_dict['rouge1_precision']]
    result_dict['fused_metric'] = fuse_score(scores)

    result_dict['prediction_vs_target'] = can_tar_dict
    result_dict['prediction_vs_source'] = can_source_dict

    return result_dict

def compute_metrics_detail2(inputs, preds, preds_values, labels, result_dict = {}, mode = 'train'):

    """
        text2data
    """

    can_tar_dict, can_source_dict = {}, {}
   

    if (mode == 'train'): 
        print('Calculating SPM...')
        score = compute_spm_batch(preds, preds_values)
        result_dict['spm'] = score['spm']

        score = compute_osf_batch(preds, labels)
        result_dict['osf_precision'] = score['osf_precision']
        result_dict['osf_f1'] = score['osf_f1']
        result_dict['osf_recall'] = score['osf_recall']
        
        score = compute_sf_batch(preds, labels)
        result_dict['sf_precision'] = score['sf_precision']
        result_dict['sf_f1'] = score['sf_f1']
        result_dict['sf_recall'] = score['sf_recall']
        
        return result_dict

    print('Calculating SPM...')
    #score = compute_spm_batch(preds, input_values)
    score = compute_spm_batch(inputs, preds_values)
    result_dict['spm'] = score['spm']

    print('Calculating BLEU...')
    score = compute_bleu_batch(preds, labels)
    can_tar_dict['bleu'] = score['bleu']
    score = compute_bleu_batch(preds, inputs)
    can_source_dict['bleu'] = score['bleu']

    print('Calculating METEOR...')
    score = compute_meteor_batch(preds, labels)
    can_tar_dict['meteor'] = score['meteor']
    score = compute_meteor_batch(preds, inputs)
    can_source_dict['meteor'] = score['meteor']
 
    print('Calculating ROUGE...')
    score = compute_rouge_batch(preds, labels)
    for k, v in score.items(): can_tar_dict[k] = v
    score = compute_rouge_batch(preds, inputs)
    for k, v in score.items(): can_source_dict[k] = v

    print('Calculating BERTSCORE...')
    score = compute_bertscore_batch(preds, labels)
    for k, v in score.items(): can_tar_dict[k] = v
    score = compute_bertscore_batch(preds, inputs)
    for k, v in score.items(): can_source_dict[k] = v
    
    # fuse metric
    scores = [result_dict['spm'], can_tar_dict['rouge1_precision']]
    result_dict['fused_metric'] = fuse_score(scores)

    result_dict['prediction_vs_target'] = can_tar_dict
    result_dict['prediction_vs_source'] = can_source_dict

    return result_dict

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
    
def test_dataset(model_name, model, tokenizer, use_force_words = False, input_file = 'dataset/test_random.json', \
                 batch_size = 8, decoding_type = 'greedy', num_beam = 4, output_dir='output/', self_pred = False, \
                 dataset_name = 'wida2wl', source_prefix = '', max_len = 256, min_len = 4, source_column = 'source', \
                 target_column = 'target'):

    dataset = read_list_from_jsonl_file(input_file)

    # add prefix
    for item in dataset: 
        #item['source'] = source_prefix + item['source']
        item[source_column] = source_prefix + item[source_column]

    if (self_pred == True):
        for item in dataset:
            try:
                #item['source'] = item['source'].split(' [SEP] ')[0] + ' [SEP] ' + item['prediction'][0]
                item[source_column] = item[source_column].split(' [SEP] ')[0] + ' [SEP] ' + item['prediction'][0]
            except:
                print('Warning: The dataset contains no "prediction"!')
                pass
            
    pred_list = generate_dataset(dataset, model_name, model, tokenizer, use_force_words = use_force_words, \
                                 input_file = input_file, decoding_type = decoding_type, num_beam = num_beam, \
                                 batch_size = batch_size, max_len = max_len, min_len = min_len)

    # calculate metrics by datasets
    dart_pred_list = []
    e2e_ref_list, e2e_pred_list = [], []
    for pred, item in zip(pred_list, dataset):
        item['prediction'] = pred
        #item['new_source'] = item[source_column] + ' [SEP] ' + pred[0]

        doc = wordpunct_tokenize(pred[0].lower())
        dart_pred_list.append(' '.join(token for token in doc if token.strip() != ''))
        e2e_pred_list.append(pred[0])

        try:
            if (type(item[target_column]) != list):
                e2e_ref_list.append(item[target_column])
                
            else:
                for target in item[target_column]: 
                    e2e_ref_list.append(target)
            e2e_ref_list.append('')
        except: pass

    output_file = input_file.replace('.json', '')
    if (use_bad_words == True): output_file += '_masked'
    output_file += '_pred.json'
    
    write_list_to_jsonl_file(output_file, dataset, 'w')

    if (dataset_name == 'dart'):

        output_file = output_file.replace('.json', '') + '_dart.txt'
        write_list_to_text_file(output_file, dart_pred_list, 'w')

        # save to "dart_eval" folder
        _, file_name = os.path.split(output_file)
        output_file = 'dart_eval/example/' + file_name
        output_folder = 'dart_eval/example/'
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        
        write_list_to_text_file(output_file, dart_pred_list, 'w')

        import subprocess
        process = subprocess.Popen(['bash', 'run_eval_on_dart.sh'], stdout=subprocess.PIPE, cwd='dart_eval')

        output, error = process.communicate()
        print('evaluation output: ', output.decode())
        

    if (dataset_name == 'e2e_nlg'):

        output_file1 = output_file.replace('.json', '') + '_e2e_ref.txt'
        write_list_to_text_file(output_file1, e2e_ref_list, file_access = 'w')

        _, file_name1 = os.path.split(output_file1)
        output_file1 = 'e2e_eval/example/' + file_name1
        if not os.path.exists('e2e_eval/example/'): os.makedirs('e2e_eval/example/')
        write_list_to_text_file(output_file1, e2e_ref_list, 'w')

        output_file2 = output_file.replace('.json', '') + '_e2e_pred.txt'
        write_list_to_text_file(output_file2, e2e_pred_list, file_access = 'w')

        _, file_name2 = os.path.split(output_file2)
        output_file2 = 'e2e_eval/example/' + file_name2
        if not os.path.exists('e2e_eval/example/'): os.makedirs('e2e_eval/example/')
        write_list_to_text_file(output_file2, e2e_pred_list, 'w')

        import subprocess
        process = subprocess.Popen(['python', 'measure_scores.py', 'example/' + file_name1, 'example/' + file_name2], \
                                   stdout=subprocess.PIPE, cwd='e2e_eval')
        output, error = process.communicate()
        print('evaluation output: ', output.decode())

    if (dataset_name == 'wida2wl'):
    
        inputs = []
        best_preds = []
        labels = []
        input_values = []
    
        for i, pred, item in zip(range(0, len(dataset)), pred_list, dataset):
            sys.stdout.write('Checking item: %d/%d \t Model: %s \r' % (i + 1, len(dataset), model_name))
            #sys.stdout.flush()

            #inp = item['source'] # source prefix?
            inp = remove_prefix(item[source_column], source_prefix)
            
            values = extract_values([inp], dataset_name = dataset_name)[0]
           
            label = item[target_column]

            # get best pred by source
            best_pred = ''
            if (decoding_type != 'greedy' and decoding_type != 'multinomial'):
                best_pred,_ = get_best_candidate_by_source(pred, inp, values)
            else:
                try: best_pred = pred[0]
                except: pass

            inputs.append(inp)
            best_preds.append(best_pred)
            labels.append(label)
            input_values.append(values)
        
        result_dict = {}
        result_dict['checked_file'] = input_file
        result_dict['model_name'] = model_name
        result_dict['decoding_type'] = decoding_type
        result_dict['use_force_words'] = use_force_words
        result_dict['use_bad_words'] = use_bad_words
        result_dict = compute_metrics_detail(inputs, input_values, best_preds, labels, result_dict, \
                        dataset_name = dataset_name, mode = 'test')                             
        result_file = output_file.replace('.json', '') + '_result.json'
        write_single_dict_to_json_file(result_file, result_dict, file_access = 'w')
        print('evaluation output: ', result_dict)


def fuse_score(scores):

    filtered_scores = []
    if (len(scores) == 1): return scores[0]
    
    for score in scores:
        if (type(score) is list): score = score[0]
        if (score > 0): filtered_scores.append(score)
   
    return statistics.harmonic_mean(filtered_scores)


def get_best_candidate_by_source(can_list, source, values, metric_list = ['rouge', 'rep', 'spm']):

    best_can = ''
    best_score = 0
   
    for can in can_list:
        scores =  []
        for metric in metric_list:
            if (metric == 'rouge'):
                scores.append(compute_spm_single(can, source)['rouge1_precision']) # error "source", pls check!!!
            if (metric == 'rep'):
                scores.append(1 - compute_repetition_single(can)['rep'])
            if (metric == 'spm'):
                scores.append(compute_spm_single(can, values)['spm'])
            
        score = fuse_score(scores)
        if (score > best_score):
            best_score = score
            best_can = can

    return best_can, best_score
            
def get_best_candidate_by_target(can_list, target, metric_list = ['rouge', 'rep', 'spm']):
    
    best_can = ''
    best_score = 0
    
    for can in can_list:
        scores =  []
        for metric in metric_list:
            if (metric == 'rouge'):
                scores.append(compute_spm_single(can, target)['rouge1_precision'])
            if (metric == 'rep'):
                scores.append(1 - compute_repetition_single(can)['rep'])
            if (metric == 'spm'):
                scores.append(compute_spm_single(can, values)['spm'])
                
        score = fuse_score(scores)
        if (score > best_score):
            best_score = score
            best_can = can

    return best_can, best_score


def convert_folder_name(folder):
    return folder.replace('.','_').replace('/','_')

def get_sentence_list(text):
    """
        get sentence list from text
            text: string - a given text
            return: list - a list of sentences
    """
    sen_list = []
    sen_list  = nltk.sent_tokenize(text)
    sen_list = [s.strip() for s in sen_list if s.strip() != '']
    
    # spaCy
    '''doc = nlp(text)
    for sent in doc.sents:
        sent_strip = ' '.join(x.strip() for x in sent.text.split() if x.strip() != '')
        sen_list.append(sent_strip)
    sen_list = [x.strip() for x in sen_list if x.strip() != '']'''
    
    return sen_list

def optimize_target(source, target, dataset_name = 'wida2wl'):

    sent_list = get_sentence_list(target)           
    values = list(set(extract_values([source], dataset_name)[0]))
    #print('values: ', values)fextract_values
    
    matched_values, matched_sents = [], []

    for sent in sent_list:
        for value in values:
            if (value in sent and value not in matched_values):
                matched_values.append(value)
                if (sent not in matched_sents):
                    matched_sents.append(sent)
    #print('matched_values: ', matched_values)
    if (len(matched_values) != len(values)): return target
    return ' '.join(s for s in matched_sents)


def add_target(current_list, source, target, pred, source_column = 'source', target_column = 'target'):

    flag = False

    for item in current_list:
        if (item[source_column] == source):
            flag = True
            if (len(pred) < len(item[target_column])):
                item[target_column] = pred
                current_list.append(item)

            #if (pred == item[target_column]): current_list.remove(item)
            break

    if (flag == False):
        current_list.append({source_column: source, target_column: pred, 'old_target': target})
    
    return current_list
        

def create_train_set(opt_d2t_pred_list, opt_t2d_pred_list, train_list, current_list = [], \
                     source_column = 'source', target_column = 'target', dataset_name = 'wida2wl'):

    for opt_d2t_pred, opt_t2d_pred, item in zip(opt_d2t_pred_list, \
                                                         opt_t2d_pred_list, train_list):
        
        source = item[source_column]
        target = item[target_column]
        
        d2t_values = extract_values([source], dataset_name = dataset_name)[0]
        
        opt_d2t_pred = opt_d2t_pred[0]
        opt_t2d_pred = opt_t2d_pred[0]
        opt_t2d_values = extract_values([opt_d2t_pred], dataset_name = dataset_name)[0]

        # value order (very rare cases)
        '''reg_str = '\s.*\s'.join(x for x in d2t_values)
        order = re.search(reg_str, opt_d2t_pred)
        if order: order = True
        else: order = False'''
 
        con1 = False
        if (len(opt_d2t_pred) < len(target) and compute_spm_single(opt_d2t_pred, d2t_values)['spm'] == 1):
            con1 = True
        
        con2 = False
        if (t2d_opt_metric == 'osf' and compute_osf_single(opt_t2d_pred, source)['osf_precision'] == 1):
            con2 = True
                   
        if (t2d_opt_metric == 'spm' and compute_spm_single(target, opt_t2d_values)['spm'] == 1):
            con2 = True
            
        if (t2d_opt_metric == 'sf' and compute_sf_single(opt_t2d_pred, source)['sf_precision'] == 1):
            con2 = True
        
        if (con1 == True and con2 == True):
            current_list = add_target(current_list, source, target, opt_d2t_pred, \
                                        source_column = source_column, target_column = target_column)
                                              
    '''for item in current_list: 
        print(item)
        print('---------------')'''
     
    print('current_list: ', len(current_list))

    # take out predictions from the previous inference
    for item in current_list:
        try: del item['prediction']
        except: pass
            
    # remove repetitions    
    current_list = [dict(t) for t in {tuple(d.items()) for d in current_list}]
    
    return current_list


def save_optimized_data(d2t_list, dataset_name, source_column = 'source', target_column = 'target'):

    # load optimized data
    output = 'dataset/' + dataset_name + '/'
    output = output.replace('//', '/')
    output_file = output + 'data_optimized.json'

    new_d2t_list = []
    if not os.path.exists(output):
        os.makedirs(output)
    else:
        new_d2t_list = read_list_from_jsonl_file(output_file)

    for item in d2t_list:
        try:
            source = item[source_column] + ' [SEP] ' + item['old_target']
            target = item[target_column]
            new_d2t_list.append({source_column: source, target_column: target})   
        except:
            pass

    # remove repetitions
    new_d2t_list = [dict(t) for t in {tuple(d.items()) for d in new_d2t_list}]
        
    write_list_to_jsonl_file(output_file, new_d2t_list, file_access = 'w')
    print('New data is already saved!')



def merge_data(list1, list2, source_column = '', target_column = ''):

    #if (len(list1) > len(list2)): return False
    
    for item2 in list2:
        flag = False
        for item1 in list1:
            if (item2[source_column] == item1[source_column] and item2[target_column] == item1[target_column]):
                flag = True
                break
        
        if (flag == False):
            list1.append(item2)
    
    return list1

def self_train(model_name, model, tokenizer, data, use_force_words = False, use_fuse_loss = False, \
               decoding_type = 'greedy', num_beam = 4, batch_size = 4, test_batch_size = 16, max_len = 256, \
               min_len = 4, train_epochs = 1, self_train_epochs = 3, output_dir = 'output/', source_column = 'source', \
               target_column = 'target', load_trained = False, d2t_model_path = '', t2d_model_path = '', dataset_name = 'wida2wl', \
               train_percent = 10, merge_new_data = True, self_train_t2d = True, same_data = True, \
               no_self_mem = False, same_data_type = 1, same_data_size = True):

    """
        the D2T and T2D models must be trained first in 1 epoch
    """

    import copy
    
    if (no_self_mem == True):

        best_model1 = copy.deepcopy(model)
        train_loss1 = 1
        best_epoch = -1
        
        train_data1, val_data1 = data['d2t']['train'], data['d2t']['val']
        
        if (same_data == True and same_data_type == 2): # no self-mem 2
            output_dir1 = output_dir + '/' + dataset_name + '/data2text/no_self_mem2'
            output_dir1 = output_dir1.replace('//', '/')

            n_examples =  int((train_percent/100)*len(train_data1['train']))
            print('n_examples: ', n_examples)
            
            train_sub_data1 = {'train': datasets.Dataset.from_dict(train_data1['train'][:n_examples])}
            train_output1 = train(model_name, model, tokenizer, train_sub_data1, val_data1, num_train_epochs = train_epochs, \
                          batch_size = batch_size, output_dir = output_dir1, generation_max_len = max_len, train_type = 'd2t')
            return 
            
        
        for i in range(1, train_epochs + 1):

            if (train_percent == 100):
                train_sub_data1 = train_data1
            else:
                if (same_data == False): # no self-mem 3
                    seed = random.randint(1, 42)
                    train_sub_data1 = train_data1['train'].train_test_split(test_size=train_percent/100, seed=seed) 
                    train_sub_data1['train'] = train_sub_data1['test']
                    train_sub_data1.pop('test')
                    
                    #return
                else: # no self-mem 1
                    n_examples =  int((train_percent/100)*len(train_data1['train']))
                    print('n_examples: ', n_examples)
                    train_sub_data1 = {'train': datasets.Dataset.from_dict(train_data1['train'][(i-1)*n_examples:i*n_examples])}
                    print(type(train_sub_data1))
                    
            print('train_sub_data1: ', len(train_sub_data1))
            output_dir1 = ''
            if (same_data == False):
                output_dir1 = output_dir + '/' + dataset_name + '/data2text/no_self_mem3/epoch_' + str(i)
            else:
                output_dir1 = output_dir + '/' + dataset_name + '/data2text/no_self_mem1/epoch_' + str(i)
            output_dir1 = output_dir1.replace('//', '/')

            model.train()
            train_output1 = train(model_name, model, tokenizer, train_sub_data1, val_data1, num_train_epochs = 1, \
                          batch_size = batch_size, output_dir = output_dir1, generation_max_len = max_len, train_type = 'd2t')
            
            best_train_loss1 = 1 - train_output1.best_metric

            if (best_train_loss1 < train_loss1):     
                train_loss1 = best_train_loss1
                best_model1 = copy.deepcopy(model)
                best_epoch = i
            model = copy.deepcopy(best_model1)
            best_model1.to(device_cpu) 
            
            torch.cuda.empty_cache()
            gc.collect()
            
        print('best_train_loss: ', best_train_loss1)
        print('best_epoch: ', str(best_epoch))
        print('same_data_type: ', same_data_type)
        
        # keep the best model only
        for j in range(1, train_epochs + 1):
            if (j == best_epoch): continue
            del_dir1 = output_dir + '/' + dataset_name + '/data2text/no_self_mem3/epoch_' + str(j) 
            try: shutil.rmtree(del_dir1)
            except: pass    
            
            del_dir1 = output_dir + '/' + dataset_name + '/data2text/no_self_mem1/epoch_' + str(j) 
            try: shutil.rmtree(del_dir1)
            except: pass    
        return
            

    t2d_model = copy.deepcopy(model) # duplicate model
    t2d_model.to(device)

    # load t2d data
    train_data2, val_data2 = data['t2d']['train'], data['t2d']['val']
    if (train_percent == 100):
        train_sub_data2 = train_data2
    else:
        train_sub_data2 = train_data2['train'].train_test_split(test_size=train_percent/100, seed = 42) 
        train_sub_data2['train'] = train_sub_data2['test']
        train_sub_data2.pop('test')
    print('train_sub_data2: ', len(train_sub_data2['train']))
    output_dir2 = output_dir + '/' + dataset_name + '/text2data'
    output_dir2 = output_dir2.replace('//', '/')

    # load d2t data
    train_data1, val_data1 = data['d2t']['train'], data['d2t']['val']
    if (train_percent == 100):
        train_sub_data1 = train_data1
    else:
        train_sub_data1 = train_data1['train'].train_test_split(test_size=train_percent/100, seed = 42) 
        train_sub_data1['train'] = train_sub_data1['test']
        train_sub_data1.pop('test')
    print('train_sub_data1: ', len(train_sub_data1['train']))
    output_dir1 = output_dir + '/' + dataset_name + '/data2text'
    output_dir1 = output_dir1.replace('//', '/')

    # load models
    train_output1_best_metric, train_output2_best_metric = 0, 0
    if (load_trained == True):
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(d2t_model_path)
            t2d_model = AutoModelForSeq2SeqLM.from_pretrained(t2d_model_path)
        except:
            print('Please define "d2t_model_path" and "t2d_model_path" correctly!')
            return
    else:
        # train from the beginning
        train_output2 = train(model_name, t2d_model, tokenizer, train_sub_data2, val_data2, num_train_epochs = train_epochs, \
                          batch_size = batch_size, output_dir = output_dir2, generation_max_len = max_len, train_type = 't2d')
        train_output2_best_metric = train_output2.best_metric
        del train_output2
        
        train_output1 = train(model_name, model, tokenizer, train_sub_data1, val_data1, num_train_epochs = train_epochs, \
                          batch_size = batch_size, output_dir = output_dir1, generation_max_len = max_len, train_type = 'd2t')
        #print('train_output: ', train_output, dir(train_output))
        #train_loss = 1 - train_output.best_metric
        train_output1_best_metric = train_output1.best_metric
        del train_output1
        
    
    # get d2t train list
    d2t_train_list = []
    for x in train_data1['train']:
        # input_ids, attention_mask, labels
        source = tokenizer.decode([y for y in x['input_ids'] if y not in [-100]])
        source = source.replace('<pad>','').replace('<s>','').replace('</s>','').strip()
        target = tokenizer.decode([y for y in x['labels'] if y not in [-100]])
        target = target.replace('<pad>','').replace('<s>','').replace('</s>','').strip()
        d2t_train_list.append({source_column:source, target_column:target})

    # self-train
    best_time1, best_time2 = -1, -1
    train_loss1, train_loss2 = 1, 1

    best_model1 = copy.deepcopy(model) 
    best_model2 = copy.deepcopy(t2d_model)
    
    torch.cuda.empty_cache()
    gc.collect()

    d2t_current_list, t2d_current_list, new_d2t_list = [], [], []
    for i in range(1, self_train_epochs + 1):

        print('***********************************************')
        print('***** SELF-TRAIN --- time: ' + str(i) + ' *****')
        
        model = copy.deepcopy(best_model1) 
        t2d_model = copy.deepcopy(best_model2)

        model.to(device)
        model.eval()

        t2d_model.to(device)
        t2d_model.eval()

        # get random examples from the train set
        n_examples = int(len(d2t_train_list)*(train_percent/100))
        print('n_examples: ', n_examples)
        
        if (same_data == True):
            d2t_train_list1 = d2t_train_list[(i-1)*n_examples:i*n_examples]
        else:
            d2t_train_list1 = random.sample(d2t_train_list, n_examples)

        # data to text predictions
        d2t_pred_list = generate_dataset(d2t_train_list1, model_name, model, tokenizer, use_force_words = use_force_words, \
                                         decoding_type = decoding_type, num_beam = num_beam, batch_size = test_batch_size, \
                                         max_len = max_len, min_len = min_len, source_column = source_column)
        print('d2t_pred_list: ', d2t_pred_list[0], len(d2t_pred_list))

        # text to data predictions
        t2d_train_list = [{source_column:pred[0]} for pred in d2t_pred_list]
        t2d_pred_list = generate_dataset(t2d_train_list, model_name, t2d_model, tokenizer, use_force_words = use_force_words, \
                                         decoding_type = decoding_type, num_beam = num_beam, batch_size = test_batch_size, \
                                         max_len = max_len, min_len = min_len, source_column = source_column)
        print('t2d_pred_list: ', t2d_pred_list[0], len(t2d_pred_list))
        
        # optimize data to text predictions
        opt_d2t_pred_list = [[optimize_target(item[source_column], pred[0])] for pred, item in zip(d2t_pred_list, d2t_train_list1)]  
        print('opt_d2t_pred_list: ', opt_d2t_pred_list[0], len(opt_d2t_pred_list))
        
        opt_train_list = [{source_column:pred[0]} for pred in opt_d2t_pred_list]
        print('opt_train_list: ', opt_train_list[0], len(opt_train_list))
        
        # optimize text to data predictions
        opt_t2d_pred_list = generate_dataset(opt_train_list, model_name, t2d_model, tokenizer, use_force_words = use_force_words, \
                                             decoding_type = decoding_type, num_beam = num_beam, batch_size = test_batch_size, \
                                             max_len = max_len, min_len = min_len, source_column = source_column)
        print('opt_t2d_pred_list: ', opt_t2d_pred_list[0], len(opt_t2d_pred_list))

        # create new d2t train set
        d2t_current_list = create_train_set(opt_d2t_pred_list, opt_t2d_pred_list, \
                                            d2t_train_list1, d2t_current_list, source_column = source_column, \
                                            target_column = target_column, dataset_name = dataset_name)
        
        # end training conditions
        if (len(d2t_current_list) == 0):
            print('There has no new data to train!')
            return
        if (len(d2t_current_list)//batch_size < 2):
            print('There has not enough new data to train!')
            return 
        print('first item: ', d2t_current_list[0])

        # save optimized data for extended training
        save_optimized_data(d2t_current_list, dataset_name)

        # take out old targets + predictions from previous inference
        for item in d2t_current_list:
            try: del item['old_target'] 
            except: pass
            try: del item['prediction']
            except: pass

        # add new data to reduce catastrophic forgetting
        if (merge_new_data == True):                       
            if (same_data_size == True):
                d2t_current_list += d2t_train_list1[0:len(d2t_train_list1) - len(d2t_current_list)]
                
                #d2t_current_list += random.sample(d2t_train_list1, len(d2t_train_list1) - len(d2t_current_list))
                #d2t_current_list += random.sample(d2t_train_list1, len(d2t_train_list1) - len(d2t_current_list))
                #d2t_current_list = merge_data(d2t_current_list, d2t_train_list1, source_column = source_column, target_column = target_column)
                '''if (len(d2t_current_list) > n_examples):
                    #d2t_current_list = random.sample(d2t_current_list, n_examples)
                    d2t_current_list = d2t_current_list[0: n_examples]'''
            else:
                d2t_current_list += d2t_train_list1
        
        # convert to t2d train set
        t2d_current_list = [{source_column: item[target_column], target_column: item[source_column]} for item in d2t_current_list]

        # load new data for d2t and t2d
        d2t_train_data = load_data2(tokenizer, batch_size, max_len, max_len, d2t_current_list, \
                                    source_column = source_column, target_column = target_column)

        t2d_train_data = load_data2(tokenizer, batch_size, max_len, max_len, t2d_current_list, \
                                    source_column = source_column, target_column = target_column)

        # train new d2t data with a single epoch
        model.train()                         
        self_dir1 = output_dir1 + '/self_train_' + str(i) + '/'
        self_train_output1 = train(model_name, model, tokenizer, d2t_train_data, val_data1, num_train_epochs = 1, \
                                  batch_size = batch_size, output_dir = self_dir1, generation_max_len = max_len, train_type = 'd2t')
        self_train_loss1 = 1 - self_train_output1.best_metric

        if (self_train_loss1 < train_loss1):     
            train_loss1 = self_train_loss1
            best_time1 = i
            best_model1 = copy.deepcopy(model)
            best_model1.to(device_cpu) 
            
        # keep the best model only
        for j in range(1, self_train_epochs + 1):
            if (j == best_time1): continue
            del_dir1 = output_dir1 + '/self_train_' + str(j) + '/'
            try: shutil.rmtree(del_dir1)
            except: pass
        
        del model
        del d2t_train_data
        d2t_current_list = []
        
        # train new t2d data with a single epoch
        if (self_train_t2d == True):
            t2d_model.train()
            self_dir2 = output_dir2 + '/self_train_' + str(i) + '/'
            self_train_output2 = train(model_name, t2d_model, tokenizer, t2d_train_data, val_data2, num_train_epochs = 1, \
                                  batch_size = batch_size, output_dir = self_dir2, generation_max_len = max_len, train_type = 't2d')
            self_train_loss2 = 1 - self_train_output2.best_metric

            if (self_train_loss2 < train_loss2):     
                train_loss2 = self_train_loss2
                best_time2 = i
                best_model2 = copy.deepcopy(t2d_model)
                best_model2.to(device_cpu) 
 
            # keep the best model only
            for j in range(1, self_train_epochs + 1):
                if (j == best_time2): continue
                del_dir2 = output_dir2 + '/self_train_' + str(j) + '/'
                try: shutil.rmtree(del_dir2)
                except: pass

        del t2d_model
        del t2d_train_data
        t2d_current_list = []


        print('***********************************************')
        print('results of data2text: ')        
        print('best_train_time: ', best_time1)
        print('best_metric (self-train): ', 1 - train_loss1)
        if (load_trained == False): 
            print('best_metric (normal train in ' + str(train_epochs) + ' epoch(s)) :', train_output1_best_metric)
        print('best_train_loss: ', train_loss1)
        if (self_train_t2d == True):
            print('----------------------------------')
            print('results of text2data: ')
            print('best_train_time: ', best_time2)
            print('best_metric (self-train): ', 1 - train_loss2)
            if (load_trained == False): 
                print('best_metric (normal train in ' + str(train_epochs) + ' epoch(s)) :', train_output2_best_metric)
            print('best_train_loss: ', train_loss2)
        print('***********************************************')
        
        torch.cuda.empty_cache()
        gc.collect()


def create_bad_words(args):
    
    bad_words = []
    if (use_bad_words == True):
        bad_words = tokenizer('XXX', add_special_tokens = False).input_ids
        bad_words = [ids for ids in bad_words if ids != -100]
        print('bad_words: ', bad_words)
    else:
        bad_words = []
    return bad_words

def main(args):

    base_setting(args)
        
    global tokenizer # for compute_metrics()
    global use_bad_words
    use_bad_words = (args.use_bad_words == 1)
    
    global bad_words

    #global seed
    #seed = args.seed
   
    global use_force_words
    use_force_words = (args.use_force_words == 1)

    global use_fuse_loss
    use_fuse_loss = (args.use_fuse_loss == 1)
    
    global infer_multi_thread
    infer_multi_thread = (args.infer_multi_thread == 1)
 
    global dataset_name
    dataset_name = args.dataset_name

    global source_prefix
    source_prefix = args.source_prefix

    global eval_metric
    eval_metric = args.eval_metric
    
    global t2d_opt_metric
    t2d_opt_metric = args.t2d_opt_metric

    if (args.mode == 'train'):
    
        start = time.time()
    
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case = False, add_prefix_space = True)
        bad_words = create_bad_words(args)
        
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        model.to(device)

        train_data, val_data = load_data(tokenizer, args.batch_size, args.encoder_max_length, \
                                         args.decoder_max_length, train_file = args.train_file, \
                                         val_file = args.var_file, source_column = args.source_column, \
                                         target_column = args.target_column, source_prefix = source_prefix)
                                         
        saved_path = args.output_dir + '/' +  convert_folder_name(args.model_name)
        saved_path = saved_path.replace('//', '/')

        train(args.model_name, model, tokenizer, train_data, val_data, num_train_epochs = args.epoch, \
              batch_size = args.batch_size, output_dir = saved_path, generation_max_len = args.decoder_max_length, \
              dataset_name = dataset_name)
        
        end = time.time()
        print("Traning time in seconds: ", (end - start))

    elif (args.mode == 'self_train'):

        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case = False, add_prefix_space = True)
        bad_words = create_bad_words(args)
        
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        model.to(device)
        
        #model = model.to_bettertransformer() 
        
        # load data for d2t
        train_data1, val_data1 = load_data(tokenizer, args.batch_size, args.encoder_max_length, \
                                         args.decoder_max_length, train_file = args.train_file, \
                                         val_file = args.var_file, source_column = args.source_column, \
                                         target_column = args.target_column, source_prefix = source_prefix)

        # load data for t2d
        train_data2, val_data2 = load_data(tokenizer, args.batch_size, args.encoder_max_length, \
                                         args.decoder_max_length, train_file = args.train_file, \
                                         val_file = args.var_file, source_column = args.target_column, \
                                         target_column = args.source_column, source_prefix = source_prefix)

        data = {}
        data['d2t'] = {'train': train_data1,  'val': val_data1}
        data['t2d'] = {'train': train_data2,  'val': val_data2}

        saved_path = args.output_dir + '/' +  convert_folder_name(args.model_name)
        saved_path = saved_path.replace('//', '/')

        load_trained = (args.load_trained == 1)
        merge_new_data = (args.merge_new_data == 1)
        self_train_t2d = (args.self_train_t2d == 1)
        same_data = (args.same_data == 1)
        same_data_size = (args.same_data_size == 1)
        no_self_mem = (args.no_self_mem == 1)
        same_data_type = args.same_data_type
        
        self_train(args.model_name, model, tokenizer, data, use_force_words = use_force_words, \
                   use_fuse_loss = use_fuse_loss, decoding_type = args.decoding_type, batch_size = args.batch_size, \
                   test_batch_size = args.test_batch_size, max_len = args.decoder_max_length, \
                   min_len = args.decoder_min_length, train_epochs = args.epoch, self_train_epochs = args.self_epoch, \
                   output_dir = saved_path, source_column = args.source_column, target_column = args.target_column, \
                   load_trained = load_trained, d2t_model_path = args.d2t_model_path, t2d_model_path = args.t2d_model_path, \
                   dataset_name = dataset_name, train_percent = args.train_percent, merge_new_data = merge_new_data, \
                   self_train_t2d = self_train_t2d, same_data = same_data, \
                   no_self_mem = no_self_mem, same_data_type = same_data_type, same_data_size = same_data_size)

        
        end = time.time()
        print("Self-traning time in seconds: ", (end - start))
        
    elif (args.mode == 'test'):

        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case = False, add_prefix_space = True)
        bad_words = create_bad_words(args)
        
        model = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir)
        model.to(device)
        model.eval()
        
        self_pred = (args.self_pred == 1)
        
        test_dataset(args.model_name, model, tokenizer, use_force_words = use_force_words, input_file = args.test_file, \
                     batch_size = args.test_batch_size, decoding_type = args.decoding_type, output_dir = args.output_dir, \
                     self_pred = self_pred, dataset_name = dataset_name, source_prefix = source_prefix, \
                     max_len = args.decoder_max_length, min_len = args.decoder_min_length, \
                     source_column = args.source_column, target_column = args.target_column)
        end = time.time()
        print("Test time in seconds: ", (end - start))

    elif (args.mode == 'generate'):

        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case = False, add_prefix_space = True)
        bad_words = create_bad_words(args)
        
        model = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir)
        model.to(device)
        model.eval()

        generate_dataset([], args.model_name, model, tokenizer, use_force_words = use_force_words, \
                         batch_size = args.test_batch_size, input_file = args.test_file, decoding_type = args.decoding_type, \
                         source_prefix = source_prefix, max_len = args.decoder_max_length, min_len = args.decoder_min_length, \
                         dataset_name = dataset_name, infer_max_workers = args.infer_max_workers)
        end = time.time()
        print("Inference time in seconds: ", (end - start))
    else:
        print('Oooopppps! You do not set any working mode.')

def base_setting(args):

    #args.seed = getattr(args, 'seed', 42)    
    args.encoder_max_length = getattr(args, 'encoder_max_length', 256)
    args.encoder_min_length = getattr(args, 'encoder_min_length', 4)
    args.decoder_max_length = getattr(args, 'decoder_max_length', 256)
    args.decoder_min_length = getattr(args, 'decoder_min_length', 4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument('--mode', type=str, default='train') # or test
    parser.add_argument('--train_file', type=str, default='dataset/train_random.json') 
    parser.add_argument('--var_file', type=str, default='dataset/validation_random.json')
    parser.add_argument('--test_file', type=str, default='dataset/test_random.json') 
    parser.add_argument('--model_name', type=str, default='facebook/bart-base')
    parser.add_argument('--decoding_type', type=str, default='greedy')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--use_force_words', type=int, default=0)
    parser.add_argument('--use_fuse_loss', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--self_epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--use_bad_words', type=int, default=0)
    parser.add_argument('--self_pred', type=int, default=0)
    parser.add_argument('--source_column', type=str, default='source')
    parser.add_argument('--target_column', type=str, default='target')
    parser.add_argument('--t2d_model_path', type=str, default='')
    parser.add_argument('--d2t_model_path', type=str, default='')
    parser.add_argument('--load_trained', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='wida2wl')
    parser.add_argument('--train_percent', type=int, default=10)
    parser.add_argument('--source_prefix', type=str, default='summarize# ')
    parser.add_argument('--merge_new_data', type=int, default=1)
    parser.add_argument('--self_train_t2d', type=int, default=1)
    parser.add_argument('--same_data', type=int, default=1)
    parser.add_argument('--eval_metric', type=str, default='eval_meteor')
    parser.add_argument('--t2d_opt_metric', type=str, default='osf')
    parser.add_argument('--no_self_mem', type=int, default=0)
    parser.add_argument('--same_data_type', type=int, default=1)
    parser.add_argument('--infer_multi_thread', type=int, default=0)
    parser.add_argument('--infer_max_workers', type=int, default=4)
    parser.add_argument('--same_data_size', type=int, default=1)
    
    args = parser.parse_args()
    main(args)
    



    
   
    
    
