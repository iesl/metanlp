# Self-Supervised Meta-Learning for Few-Shot Natural Language Classification Tasks
Code for training the meta-learning models and fine-tuning on downstream tasks.

Paper: [Self-Supervised Meta-Learning for Few-Shot Natural Language Classification Tasks](https://arxiv.org/pdf/2009.08445.pdf)

## Trained Models
- Hybrid-SMLMT: https://drive.google.com/file/d/1k5WJl-rZ8ks__PTPdoS9ibLmY2ifP6H5/view?usp=sharing
- SMLMT: https://drive.google.com/file/d/1DkQShmdf1BNLdWOL4JvdD_j0aERHZgr2/view?usp=sharing
- LEOPARD: https://drive.google.com/file/d/1JybValybM8sqJHKPAKh9g6VxUTsqfGEe/view?usp=sharing

## Dependencies
- Python version 3.6.6 or higher
- Tensorflow version 1.12.0 (higher versions might not work)
- Numpy 1.16.4 or higher
- six 1.12.0

`pip install -r requirements.txt` should install required depedencies. It is recommended to use a conda environment and make sure to use the pip installed in the environment.

## Fine-Tuning
A script is provided to run fine-tuning for a target task, by default it runs fine-tuning on CoNLL. 
The script will download all necessary data and models, 
note that in case downloads fail please download the files manually using the links. 

Fine-tuning runs on a single GPU and typically takes a few minutes.

Run the script as: `./run_finetune.sh`

Modify the following parameters in run_finetune.sh to run on a different task, or a different k-shot, or a different file split for the task:

- TASK_NAME: should be one of:  `airline, conll, disaster, emotion, political_audience, political_bias, political_message, rating_books, rating_dvd, rating_electronics, rating_kitchen, restaurant, scitail, sentiment_books, sentiment_dvd, sentiment_electronics, sentiment_kitchen`
- DATA_DIR: path to data directory (eg., `data/leopard-master/data/tf_record/${TASK_NAME}`)
- F: file train split id, should be in [0, 9]
- K: which k-shot experiment to run, should be in {4, 8, 16, 32}
- N: number of classes in the task (see paper if not known)

So, the fine-tuning run command to run on a particular split for a task is: `./run_finetune.sh TASK_NAME F K N`

To change the output directory or other arguments, edit the corresponding arguments in run_finetune.sh

Hyper-parameters for Hybrid-SMLMT
- K = 4:  
--num_train_epochs=150\*N  
--train_batch_size=4\*N

- K = 8:  
--num_train_epochs=175\*N  
--train_batch_size=8\*N

- K = 16:  
--num_train_epochs=200\*N  
--train_batch_size=4\*N

- K = 32:  
--num_train_epochs=100\*N  
--train_batch_size=8\*N  
  
### Data for fine-tuning
The data for the fine-tuning tasks can be downloaded from https://github.com/iesl/leopard 
  
### Fine-tuning on other tasks
To run fine-tuning on a different task than provided with the code, you will need to set up the train and test data for the task in a tf_record file, similar to the data for the provided tasks.

The features in the tf_record are:
```python
name_to_features = {
      "input_ids": tf.FixedLenFeature([128], tf.int64),
      "input_mask": tf.FixedLenFeature([128], tf.int64),
      "segment_ids": tf.FixedLenFeature([128], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
  }
```
where:
- input_ids: the input sequence tokenized using the BERT tokenizer
- input_mask: mask of 0/1 corresponding to the input_ids
- segment_ids: 0/1 segment ids following BERT
- label_ids: classification label

Note that the above features are same as that used in the code of BERT fine-tuning for classification, 
so code in the BERT [github repository](https://github.com/google-research/bert) can be used for creating the tf_record files.

The followiing arguments to run_classifier_pretrain.py need to be set:
- task_eval_files: train_tf_record, eval_tf_record
    - where train_tf_record is the train file for the task and eval_tf_record is the test file
- test_num_labels: number of classes in the task

## Meta-Training
This requires large training time and typically should be run on multiple GPU.

SMLMT data file name should begin with "meta_pretain" and end with the value of N for the tasks in that file (on file per N), for example "meta_pretrain_3.tf_record" for 3-way tasks. The training code will take `train_batch_size` many examples at a time starting from the beginning of the files (without shuffling) and treat that as one task for training.

Meta-training can be run using the following command:

```bash
python run_classifier_pretrain.py \
    --do_train=true \
    --task_train_files=${TRAIN_FILES} \
    --num_train_epochs=1 \
    --save_checkpoints_steps=5000 \
    --max_seq_length=128 \
    --task_eval_files=${TASK_EVAL_FILES} \
    --tasks_per_gpu=1 \
    --num_eval_tasks=1 \
    --num_gpus=4 \
    --learning_rate=1e-05 \
    --train_lr=1e-05 \
    --keep_prob=0.9 \
    --attention_probs_dropout_prob=0.1 \
    --hidden_dropout_prob=0.1 \
    --SGD_K=1 \
    --meta_batchsz=80 \
    --num_batches=8 \
    --train_batch_size=90 \
    --min_layer_with_grad=0 \
    --train_word_embeddings=true \
    --use_pooled_output=true \
    --output_layers=2 \
    --update_only_label_embedding=true \
    --use_euclidean_norm=false \
    --label_emb_size=256 \
    --stop_grad=true \
    --eval_batch_size=90 \
    --eval_examples_per_task=2000 \
    --is_meta_sgd=true \
    --data_sqrt_sampling=true \
    --deep_set_layers=0 \
    --activation_fn=tanh \
    --clip_lr=true \
    --inner_epochs=1 \
    --warp_layers=true \
    --min_inner_steps=5 \
    --average_query_every=3 \
    --weight_query_loss=true \
    --output_dir=${output_dir} \
    --pretrain_task_weight=0.5
```


## References:
Code is based on the public repository: https://github.com/google-research/bert

Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805, 2018.