#!/usr/bin/env bash

# download and extract BERT for vocab and config files
if [[ ! -d "cased_L-12_H-768_A-12" ]]
then
    echo "Downloading BERT"
    wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
    unzip cased_L-12_H-768_A-12.zip
    rm cased_L-12_H-768_A-12.zip
fi

# download and extract Hybrid-SMLMT model, since the file is large this generates a temp cookie to approve the download
# note that is this fails, manually download the file from the following link, unzip and place it in models folder
# https://drive.google.com/file/d/1k5WJl-rZ8ks__PTPdoS9ibLmY2ifP6H5/view?usp=sharing
if [[ ! -d "models/HSMLMT" ]]
then
    echo "Downloading HSMLMT"
    wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1k5WJl-rZ8ks__PTPdoS9ibLmY2ifP6H5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1k5WJl-rZ8ks__PTPdoS9ibLmY2ifP6H5" -O HSMLMT.zip
    mkdir -p models
    unzip -d models/HSMLMT HSMLMT.zip
fi

# download data
if [[ ! -d "data/leopard-master/data" ]]
then
    echo "Downloading Data"
    wget https://github.com/iesl/leopard/archive/master.zip
    unzip -d data master.zip
fi

MODEL_DIR=models/HSMLMT/model/
TASK_NAME=${1:-conll}
DATA_DIR=data/leopard-master/data/tf_record/$TASK_NAME
F=${2:-0}
K=${3:-4}
N=${4:-4}
BERT_CONFIG_FILE=cased_L-12_H-768_A-12/bert_config.json
BERT_VOCAB_PATH=cased_L-12_H-768_A-12/vocab.txt

echo ${TASK_NAME}
mkdir -p output/${TASK_NAME}\_output\_${F}\_${K}
python run_classifier_pretrain.py \
    --do_eval=true \
    --task_eval_files=$DATA_DIR/$TASK_NAME\_train_$F\_$K.tf_record,$DATA_DIR/$TASK_NAME\_eval.tf_record  \
    --warm_start_model_dir=$MODEL_DIR \
    --output_dir=output/${TASK_NAME}\_output\_${F}\_${K} \
    --max_seq_length=128 \
    --num_train_epochs=$((150*${N})) \
    --train_batch_size=$((4*${N})) \
    --eval_batch_size=$((4*${N})) \
    --test_num_labels=$N \
    --data_dir=$DATA_DIR \
    --bert_config_file=$BERT_CONFIG_FILE \
    --vocab_file=$BERT_VOCAB_PATH \
    --keep_prob=0.9 \
    --attention_probs_dropout_prob=0.1 \
    --hidden_dropout_prob=0.1 \
    --SGD_K=1 \
    --min_layer_with_grad=0 \
    --train_word_embeddings=true \
    --use_pooled_output=true \
    --output_layers=2 \
    --update_only_label_embedding=true \
    --use_euclidean_norm=false\
    --label_emb_size=256 \
    --clip_lr=true \
    --warp_layers=true \
    --train_lr=1e-5 1> output/${TASK_NAME}\_output\_${F}\_${K}/log 2>&1
