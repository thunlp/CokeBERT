#!/bin/bash


#export BACKBONE=bert-base-uncased
export BACKBONE=bert-base-uncased-2hop
export PYTHONPATH=../src:$PYTHONPATH
export HOP=2

rm outputs/finetune_coke-$BACKBONE-$HOP/*

python3 run_finetune.py \
            --output_dir outputs \
            --do_train \
            --do_lower_case \
            --data_dir ../data/finetune/fewrel/ \
            --backbone $BACKBONE \
            --neighbor_hop $HOP \
            --max_seq_length 256 \
            --train_batch_size 64 \
            --learning_rate 2e-5 \
            --num_train_epochs 16 \
            --loss_scale 128 \
            --K_V_dim 100 \
            --Q_dim 768 \
            --self_att
