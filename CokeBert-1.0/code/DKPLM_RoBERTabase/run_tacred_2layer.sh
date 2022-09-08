rm output_tacred_2layer/*

model_base=../../data/DKPLM_RoBERTabase_2layer
#model_base=ernie_base

# train
CUDA_VISIBLE_DEVICES=6,7 python3 -W ignore::UserWarning code/run_tacred_2layer.py   --do_train   --do_lower_case   --data_dir ../../data/data/tacred   --ernie_model $model_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 5.0   --output_dir output_tacred_2layer   --fp16   --loss_scale 128 --threshold 0.4 --K_V_dim 100 --Q_dim 768 --self_att --data_token tacred

# evaluate
rm output_tacred_2layer/*comb
CUDA_VISIBLE_DEVICES=6,7 python3 -W ignore::UserWarning code/eval_tacred_2layer.py   --do_eval   --do_lower_case   --data_dir ../../data/tacred   --ernie_model $model_base   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 5.0   --output_dir output_tacred_2layer   --fp16   --loss_scale 128 --threshold 0.4 --K_V_dim 100 --Q_dim 768 --self_att --data_token tacred

python3 code/score.py output_tacred_2layer
