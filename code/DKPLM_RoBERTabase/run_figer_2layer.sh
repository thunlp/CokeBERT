
model_base=../../data/DKPLM_RoBERTabase_2layer
#model_base=ernie_base
rm output_figer_2layer/*

#train
#train_batch_size == 2048 -> 1024
CUDA_VISIBLE_DEVICES=4,5 python3 -W ignore::UserWarning code/run_figer_2layer.py    --do_train   --do_lower_case   --data_dir ../../data/data/FIGER   --ernie_model  $model_base   --max_seq_length 256   --train_batch_size 1024   --learning_rate 2e-5   --num_train_epochs 6.0   --output_dir output_figer_2layer --gradient_accumulation_steps 32 --threshold 0.3 --fp16 --loss_scale 128 --warmup_proportion 0.2 --K_V_dim 100 --Q_dim 768 --self_att --data_token figer

# evaluate
#train_batch_size == 2048 -> 1024
CUDA_VISIBLE_DEVICES=4,5 python3 -W ignore::UserWarning code/eval_figer_2layer.py   --do_eval   --do_lower_case   --data_dir ../../data/data/FIGER   --ernie_model $model_base   --max_seq_length 256   --train_batch_size 1024   --learning_rate 2e-5   --num_train_epochs 6.0   --output_dir output_figer_2layer --gradient_accumulation_steps 32 --threshold 0.3 --fp16 --loss_scale 128 --warmup_proportion 0.2 --K_V_dim 100 --Q_dim 768 --self_att --data_token figer
