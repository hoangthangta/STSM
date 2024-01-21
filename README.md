# Introduction

This is a repo for our paper "Self-training from Self-memory in Data-to-text Generation". We test our self-training model over two datasets, DART and E2E NLG. To check examples of NLG datasets, we suggest you to use: https://quest.ms.mff.cuni.cz/nlg/tabgenie.

![Alt text](model.png?raw=true "STSM model")

# Dependencies
* Python >= 3.8.x 
* transformers >= 4.30.2
* torch >= 2.0.1+cu117 (not sure with previous versions)

# Training information
We train our models on "t5-base" (without prefixes), "facebook/bart-base", and "google/flan-t5-base" on Windows, using Git Bash. All self-memory training methods only rely on "t5-base". We adapt the evaluation of DART and E2E NLG from two original packages (https://github.com/Yale-LILY/dart/tree/master/evaluation and https://github.com/tuetschek/e2e-metrics). Furthermore, we cut off the evaluation on several metrics due to errors. We also offer our evaluation version, which can be found in the file **metrics.py**.

# Training parameters
Here is the list of training parameters:
* *mode*: train/test/generate
* *epoch*: the number of epochs
* *batch_size*: the size of batch
* *train_file, var_file, test_file*: URLs of training, validation, and test files
* *decoding_type*: greedy
* *output_dir*: the location of trained models
* *dataset_name*: the name of datasets (dart/e2e)
* *source_prefix*: helpful for T5 models 
* *source_column, target_column*: specify the fields of input and output in the training, validation, and test sets
* *train_percent*: the ratio of self-train data (30%).
* *merge_new_data*: merge self-memory with new data (1) or no merge self-memory with new data (0)
* *self_train_t2d*: self-train the T2D model
* *same_data*: train on the fixed data (1) or the random data (0)
* *eval_metric*: evaluation metric
* *t2d_opt_metric*: T2D optimization metric (osf = Overall Slot Filling, spm = simple phrase matching)
* *no_self_mem*: use self-memory (1) or not (0)
* *same_data_type*: use with no_self_mem = 1
* *infer_multi_thread*: use multi-threading in the inference time, (0 = No (default), 1 = Yes)
* *infer_max_workers*: use with *infer_multi_thread = 1* in the inference time, (default is 4)


# DART
### FULL TRAIN
python seq2seq.py --mode train --epoch 3 --batch_size 4 --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --decoding_type "greedy" --output_dir "output/" --dataset_name "dart" --source_prefix "" --source_column "source" --target_column "target"

### ACCELERATE FULL TRAIN
accelerate launch seq2seq.py --mode train --epoch 3 --batch_size 4 --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --decoding_type "greedy" --output_dir "output/" --dataset_name "dart" --source_prefix "" --source_column "source" --target_column "target"

### 30% FIXED DATA + WITH NEW DATA + WITH SELF-TRAIN T2D
python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --merge_new_data 1 --self_train_t2d 1 --same_data 1 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

### 30% FIXED DATA + WITH NEW DATA + WITHOUT  SELF-TRAIN T2D
python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --merge_new_data 1 --self_train_t2d 0 --same_data 1 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

### 30% FIXED DATA + WITHOUT NEW DATA + WITH SELF-TRAIN T2D
python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --merge_new_data 0 --self_train_t2d 1 --same_data 1 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

### 30% FIXED DATA + WITHOUT NEW DATA + WITHOUT  SELF-TRAIN T2D
python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --merge_new_data 0 --self_train_t2d 0 --same_data 1 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

### 30% FIXED DATA + WITHOUT SELF MEMORY: TYPE 1
accelerate launch seq2seq.py --mode self_train --epoch 3 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --same_data 1 --eval_metric "eval_meteor" --no_self_mem 1 --same_data_type 1

### 30% FIXED DATA + WITHOUT SELF MEMORY: TYPE 2
accelerate launch seq2seq.py --mode self_train --epoch 3 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --same_data 1 --eval_metric "eval_meteor" --no_self_mem 1 --same_data_type 2

### 30% RANDOM DATA + WITH NEW DATA + WITH SELF-TRAIN T2D
python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --merge_new_data 1 --self_train_t2d 1 --same_data 0 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

### 30% RANDOM DATA + WITH NEW DATA + WITHOUT  SELF-TRAIN T2D
python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --merge_new_data 1 --self_train_t2d 0 --same_data 0 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

### 30% RANDOM DATA + WITHOUT NEW DATA + WITH SELF-TRAIN T2D
python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --merge_new_data 0 --self_train_t2d 1 --same_data 0 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

### 30% RANDOM DATA + WITHOUT NEW DATA + WITHOUT  SELF-TRAIN T2D
python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --merge_new_data 0 --self_train_t2d 0 --same_data 0 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

### 30% RANDOM DATA + WITHOUT SELF MEMORY: TYPE 3
python seq2seq.py --mode self_train --epoch 3 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --same_data 0 --eval_metric "eval_meteor" --no_self_mem 1

accelerate launch seq2seq.py --mode self_train --epoch 3 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --same_data 0 --eval_metric "eval_meteor" --no_self_mem 1 

### EVALUATION (use "dart_val")
python seq2seq.py --mode test --test_file "dataset/dart/test.json" --model_name "t5-base" --output_dir "output/xxx" --source_prefix "" --decoding_type "greedy" --test_batch_size 16 --dataset_name "dart"

### GENERATE
python seq2seq.py --mode generate --test_file "dataset/dart/test.json" --model_name "t5-base" --output_dir "output/xxx" --source_prefix "" --decoding_type "greedy" --test_batch_size 16 --dataset_name "dart" 

### GENERATE with MULTITHREAD ((speed up generation)
python seq2seq.py --mode generate --test_file "dataset/dart/test.json" --model_name "t5-base" --output_dir "output/xxx" --source_prefix "" --decoding_type "greedy" --test_batch_size 16 --dataset_name "dart"  --infer_max_workers 4 --infer_multi_thread 1

# E2E NLG
### FULLTRAIN
python seq2seq.py --mode train --epoch 3 --batch_size 4 --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --decoding_type "greedy" --output_dir "output/" --dataset_name "e2e_nlg" --source_prefix "" --source_column "source" --target_column "target"

### ACCELERATE FULL TRAIN
 accelerate launch seq2seq.py --mode train --epoch 3 --batch_size 4 --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --decoding_type "greedy" --output_dir "output/" --dataset_name "e2e_nlg" --source_prefix "" --source_column "source" --target_column "target"

### 30% FIXED DATA + WITH NEW DATA + WITH SELF-TRAIN T2D
python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "e2e_nlg" --source_prefix "" --merge_new_data 1 --self_train_t2d 1 --same_data 1 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

### 30% FIXED DATA + WITH NEW DATA + WITHOUT  SELF-TRAIN T2D
python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "e2e_nlg" --source_prefix "" --merge_new_data 1 --self_train_t2d 0 --same_data 1 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

### 30% FIXED DATA + WITHOUT NEW DATA + WITH SELF-TRAIN T2D
python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --merge_new_data 0 --self_train_t2d 1 --same_data 1 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

### 30% FIXED DATA + WITHOUT NEW DATA + WITHOUT  SELF-TRAIN T2D
python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "e2e_nlg" --source_prefix "" --merge_new_data 0 --self_train_t2d 0 --same_data 1 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

### 30% FIXED DATA + WITHOUT SELF MEMORY: TYPE 1
accelerate launch seq2seq.py --mode self_train --epoch 3 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "e2e_nlg" --source_prefix "" --same_data 1 --eval_metric "eval_meteor" --no_self_mem 1 --same_data_type 1

### 30% FIXED DATA + WITHOUT SELF MEMORY: TYPE 2
accelerate launch seq2seq.py --mode self_train --epoch 3 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "e2e_nlg" --source_prefix "" --same_data 1 --eval_metric "eval_meteor" --no_self_mem 1 --same_data_type 2

### 30% RANDOM DATA + WITH NEW DATA + WITH SELF-TRAIN T2D
python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "e2e_nlg" --source_prefix "" --merge_new_data 1 --self_train_t2d 1 --same_data 0 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

### 30% RANDOM DATA+ WITH NEW DATA + WITHOUT  SELF-TRAIN T2D
python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "e2e_nlg" --source_prefix "" --merge_new_data 1 --self_train_t2d 0 --same_data 0 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

### 30% RANDOM DATA + WITHOUT NEW DATA + WITH SELF-TRAIN T2D
python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --merge_new_data 0 --self_train_t2d 1 --same_data 0 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

### 30% RANDOM DATA + WITHOUT NEW DATA + WITHOUT  SELF-TRAIN T2D
python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "e2e_nlg" --source_prefix "" --merge_new_data 0 --self_train_t2d 0 --same_data 0 --eval_metric "eval_meteor" --t2d_opt_metric "osf"

### 30% RANDOM DATA + WITHOUT SELF MEMORY: TYPE 3
python seq2seq.py --mode self_train --epoch 3 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --same_data 0 --eval_metric "eval_meteor" --no_self_mem 1

accelerate launch seq2seq.py --mode self_train --epoch 3 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/e2e_nlg/train.json" --var_file "dataset/e2e_nlg/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "e2e_nlg" --source_prefix "" --same_data 0 --eval_metric "eval_meteor" --no_self_mem 1 

### EVALUATION (use "e2e_val")
python seq2seq.py --mode test --test_file "dataset/e2e_nlg/test.json" --model_name "t5-base" --output_dir "output/xxx" --source_prefix "" --decoding_type "greedy" --test_batch_size 16 --dataset_name "e2e_nlg"

### GENERATE
python seq2seq.py --mode generate --test_file "dataset/e2e_nlg/test.json" --model_name "t5-base" --output_dir "output/xxx" --source_prefix "" --decoding_type "greedy" --test_batch_size 16 --dataset_name "e2e_nlg"

### GENERATE with MULTITHREAD (speed up generation)
python seq2seq.py --mode generate --test_file "dataset/e2e_nlg/test.json" --model_name "t5-base" --output_dir "output/xxx" --source_prefix "" --decoding_type "greedy" --test_batch_size 16 --dataset_name "e2e_nlg" --infer_max_workers 4 --infer_multi_thread 1

# OTHER SELF-TRAINING
If you already have pre-trained D2T and T2D models, you can continue to self-train them. For example, self-train on DART:

python seq2seq.py --mode self_train --epoch 1 --self_epoch 2 --batch_size 4 --use_force_words 0 --use_fuse_loss 0 --decoding_type "greedy" --train_file "dataset/dart/train.json" --var_file "dataset/dart/val.json" --model_name "t5-base" --output_dir "output/" --source_column "source" --target_column "target" --train_percent 30 --dataset_name "dart" --source_prefix "" --merge_new_data 0 --self_train_t2d 1 --same_data 1 --eval_metric "eval_meteor" --t2d_opt_metric "osf" **--load_trained 1 --d2t_model_path "url_pretrained_model" --t2d_model_path  "url_pretrained_model"**

# Minor errors
The inference time in self-training is sometimes very slow, and we are working to find out why. We do not apply "accelerate" in self-training.

# Author Information
If you have any questions, please open issues or contact:
* tahoangthang@gmail.com
* https://www.tahoangthang.com
