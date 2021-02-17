#!/bin/bash
#ernie_model
#finetune train
#bert_base => ernie_base
#graphsage


model_base=../../data/DKPLM_RoBERTabase_2layer
#model_base=ernie_base

rm output_fewrel_2layer/*


#num_train_epochs 10 -->15
CUDA_VISIBLE_DEVICES=0,1 python3 -W ignore::UserWarning code/run_fewrel_2layer.py   --do_train   --do_lower_case   --data_dir ../../data/data/fewrel/   --ernie_model $model_base   --max_seq_length 256   --train_batch_size 32 --learning_rate 2e-5   --num_train_epochs 16   --output_dir output_fewrel_2layer   --loss_scale 128 --fp16 --K_V_dim 100 --Q_dim 768 --self_att --data_token fewrel

#eval
#rm output_fewrel_new_n_CLS_comb200_rob/*_comb
CUDA_VISIBLE_DEVICES=0,1 python3 -W ignore::UserWarning code/eval_fewrel_layer.py   --do_eval   --do_lower_case   --data_dir ../../data/data/fewrel/   --ernie_model $model_base   --max_seq_length 256  --train_batch_size 16  --learning_rate 2e-5   --num_train_epochs 10   --output_dir output_fewrel_2layer   --fp16   --loss_scale 128 --K_V_dim 100 --Q_dim 768 --self_att --data_token fewrel

python3 code/score.py output_fewrel_2layer

