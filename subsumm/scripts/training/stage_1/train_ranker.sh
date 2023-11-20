#!/bin/bash

# general settings
PROJECT_NAME=rank_rt1
BART_DIR=artifacts/bart
CHECKPOINT=$BART_DIR/bart.base.pt
DATA_DIR=
LOG_INTERVAL=50
LOG_DIR=log/$PROJECT_NAME
LOG_FILE=log/logs-$PROJECT_NAME.txt
SAVE_DIR=

# general hyper-params
LR=3e-05
WARMUP=100
INITIAL_LR=1e-08
EPOCHS=5
UPDATE=1600    # 5 epochs for batch_size=8
BATCH_SIZE=1
ACCUMULATE=8

# the task-specific arguments of review_ranking_task
MARGIN=0.05
NDOCS=10  # the number of reviews to select from each collection

python subsumm/train.py --data=$DATA_DIR --bpe=gpt2 --user-dir=subsumm --bart-dir=$BART_DIR  \
--save-dir=$SAVE_DIR --restore-file=$CHECKPOINT --task=review_ranking_task \
--layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed \
--memory-efficient-fp16 --arch=ranker_base --criterion=ranking_loss --dropout=0.15 \
--attention-dropout=0.1 --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9, 0.999)" \
--adam-eps=1e-08 --clip-norm=0.1 --lr=$INITIAL_LR --max-lr=$LR --lr-scheduler=cosine \
--max-update=$UPDATE --warmup-updates=$WARMUP --max-epoch=$EPOCHS --num-workers=0 \
--reset-optimizer --required-batch-size-multiple=1 \
--reset-dataloader --reset-meters --reset-lr-scheduler \
--skip-invalid-size-inputs-valid-test --find-unused-parameters \
--sep-symb=" </s>"  --loss-type=$LOSS_TYPE  --ndocs=$NDOCS --batch-size=$BATCH_SIZE --update-freq=$ACCUMULATE --margin=$MARGIN \
--log-interval=$LOG_INTERVAL --shuffle --log-format=json --tensorboard-logdir=$LOG_DIR --ddp-backend=no_c10d | tee $LOG_FILE

