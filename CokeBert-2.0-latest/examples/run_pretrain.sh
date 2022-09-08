#!/bin/bash


export BACKBONE=bert-base-uncased
export HOP=2
export PYTHONPATH=../src:$PYTHONPATH

rm outputs/pretrain_coke-$BACKBONE-$HOP/*

python run_pretrain.py \
            --output_dir outputs \
            --data_dir ../data/pretrain \
            --backbone $BACKBONE \
            --neighbor_hop $HOP \
            --do_train \
            --max_seq_length 256 \
            --K_V_dim 100 \
            --Q_dim 768 \
            --train_batch_size 32 \
            --self_att
