#!/bin/bash

# general settings
PROJECT_NAME=verd_sentisel
BART_DIR=artifacts/bart
CHECKPOINT=$BART_DIR/bart.base.pt
DATA_DIR=
LOG_INTERVAL=50
LOG_FILE=log/logs-$PROJECT_NAME.txt
SAVE_DIR=

# general hyper-params
LR=3e-05
WARMUP=5000
EPOCHS=20
NSENTS=4
ACCUMULATE=2

# the number of reviews to select from each collection
NDOCS=10

# the task-specific arguments of sentisel_task
DATASET=amasum
TARGET=verd

python subsumm/train.py --data=$DATA_DIR --bpe=gpt2 --user-dir=subsumm --bart-dir=$BART_DIR  \
--save-dir=$SAVE_DIR --restore-file=$CHECKPOINT --task=sentisel_task \
--dataset-name=$DATASET --target-name=$TARGET \
--layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed \
--memory-efficient-fp16 --arch=sum_base --criterion=cross_entropy --dropout=0.15 \
--attention-dropout=0.1 --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9, 0.999)" \
--adam-eps=1e-08 --clip-norm=0.1 --lr=$LR --lr-scheduler=fixed \
--max-update=0 --warmup-updates=$WARMUP --max-epoch=$EPOCHS --num-workers=0 \
--reset-optimizer --required-batch-size-multiple=1 \
--reset-dataloader --reset-meters --reset-lr-scheduler \
--skip-invalid-size-inputs-valid-test --find-unused-parameters \
--sep-symb=" </s>" --max-sentences=$NSENTS --update-freq=$ACCUMULATE --ndocs=$NDOCS \
--log-interval=$LOG_INTERVAL --shuffle --log-format=json --ddp-backend=no_c10d | tee $LOG_FILE

