# Introduction

This is a repo for "Self-training from Self-memory" (STSM). We test our self-training model over two datasets, DART and E2E NLG.

# DART
## FULL TRAIN
python seq2seq.py --mode train --epoch 3 --batch_size 4 --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "facebook/bart-base" --decoding_type "greedy" --output_dir "output/" --dataset_name "dart" --source_prefix "" --source_column "source" --target_column "target"

## ACCELERATE FULL TRAIN
accelerate launch seq2seq.py --mode train --epoch 3 --batch_size 4 --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "facebook/bart-base" --decoding_type "greedy" --output_dir "output/" --dataset_name "dart" --source_prefix "" --source_column "source" --target_column "target"

## 30% DATA + WITH NEW DATA + WITH T2D
python3 seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --merge_new_data 1 --self_train_t2d 1 --same_data 1 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

## 30% DATA + WITH NEW DATA + WITHOUT T2D
python3 seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --merge_new_data 1 --self_train_t2d 0 --same_data 1 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

## 30% DATA + WITHOUT NEW DATA + WITH T2D
python3 seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --merge_new_data 0 --self_train_t2d 1 --same_data 1 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

## 30% DATA + WITHOUT NEW DATA + WITHOUT T2D
python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --merge_new_data 0 --self_train_t2d 0 --same_data 1 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

## 30% FIXED DATA + WITHOUT SELF MEMORY: TYPE 1
accelerate launch seq2seq.py --mode self_train --epoch 3 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --same_data 1 --eval_metric "eval_meteor" --no_self_mem 1 --same_data_type 1

## 30% FIXED DATA + WITHOUT SELF MEMORY: TYPE 2
accelerate launch seq2seq.py --mode self_train --epoch 3 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --same_data 1 --eval_metric "eval_meteor" --no_self_mem 1 --same_data_type 2


## 30% RANDOM DATA + WITHOUT SELF MEMORY: TYPE 3
python3 seq2seq.py --mode self_train --epoch 3 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --same_data 0 --eval_metric "eval_meteor" --no_self_mem 1

accelerate launch seq2seq.py --mode self_train --epoch 3 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --same_data 0 --eval_metric "eval_meteor" --no_self_mem 1 

## INFERENCE
python3 seq2seq.py --mode test --test_file "dataset/dart/test.json" --model_name "facebook/bart-base" --output_dir "output/xxx" -source_prefix "" --decoding_type "greedy" --test_batch_size 16 --dataset_name "dart"

## GENERATE
python3 seq2seq.py --mode generate --test_file "dataset/dart/test.json" --model_name "t5-base" --output_dir "output/xxx" --source_prefix "" --decoding_type "greedy" --test_batch_size 16 --dataset_name "dart" 

# E2E NLG
## FULLTRAIN
python seq2seq.py --mode train --epoch 3 --batch_size 4 --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "facebook/bart-base" --decoding_type "greedy" --output_dir "output/" --dataset_name "e2e_nlg" --source_prefix "" --source_column "source" --target_column "target"

## ACCELERATE FULL TRAIN
 accelerate launch seq2seq.py --mode train --epoch 3 --batch_size 4 --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "facebook/bart-base" --decoding_type "greedy" --output_dir "output/" --dataset_name "e2e_nlg" --source_prefix "" --source_column "source" --target_column "target"

## 30% DATA + WITH NEW DATA + WITH T2D
python3 seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "e2e_nlg" --source_prefix "" --merge_new_data 1 --self_train_t2d 1 --same_data 1 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

## 30% DATA + WITH NEW DATA + WITHOUT T2D
python3 seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "e2e_nlg" --source_prefix "" --merge_new_data 1 --self_train_t2d 0 --same_data 1 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

## 30% DATA + WITHOUT NEW DATA + WITH T2D
python3 seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --merge_new_data 0 --self_train_t2d 1 --same_data 1 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

## 30% DATA + WITHOUT NEW DATA + WITHOUT T2D
python3 seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 2 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "e2e_nlg" --source_prefix "" --merge_new_data 0 --self_train_t2d 0 --same_data 1 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

## 30% FIXED DATA + WITHOUT SELF MEMORY: TYPE 1
accelerate launch seq2seq.py --mode self_train --epoch 3 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "e2e_nlg" --source_prefix "" --same_data 1 --eval_metric "eval_meteor" --no_self_mem 1 --same_data_type 1

## 30% FIXED DATA + WITHOUT SELF MEMORY: TYPE 2
accelerate launch seq2seq.py --mode self_train --epoch 3 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "e2e_nlg" --source_prefix "" --same_data 1 --eval_metric "eval_meteor" --no_self_mem 1 --same_data_type 2


## 30% RANDOM DATA + WITHOUT SELF MEMORY: TYPE 3
python3 seq2seq.py --mode self_train --epoch 3 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --same_data 0 --eval_metric "eval_meteor" --no_self_mem 1

accelerate launch seq2seq.py --mode self_train --epoch 3 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "e2e_nlg" --source_prefix "" --same_data 0 --eval_metric "eval_meteor" --no_self_mem 1 

## INFERENCE
python3 seq2seq.py --mode test --test_file "dataset/e2e_nlg/test.json" --model_name "facebook/bart-base" --output_dir "output/xxx" --source_prefix "" --decoding_type "greedy" --test_batch_size 16 --dataset_name "e2e_nlg"

## GENERATE
python3 seq2seq.py --mode generate --test_file "dataset/e2e_nlg/test.json" --model_name "t5-base" --output_dir "output/xxx" --source_prefix "" --decoding_type "greedy" --test_batch_size 16 --dataset_name "e2e_nlg"

# Author Information
If you have any questions, please open issues or contact tahoangthang@gmail.com.
