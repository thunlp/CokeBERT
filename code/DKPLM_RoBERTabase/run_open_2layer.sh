
model_base=../../data/DKPLM_RoBERTabase_2layer
#model_base=ernie_base
rm output_open_2layer/*

#train
#batch 32-->16
CUDA_VISIBLE_DEVICES=2,3 python3 -W ignore::UserWarning code/run_typing_2layer.py    --do_train   --do_lower_case   --data_dir ../../data/data/OpenEntity   --ernie_model $model_base   --max_seq_length 256   --train_batch_size 16   --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir output_open_2layer --threshold 0.3 --fp16 --loss_scale 128 --K_V_dim 100 --Q_dim 768 --self_att --data_token oepn

# evaluate
#batch 32-->16
CUDA_VISIBLE_DEVICES=2,3 python3 -W ignore::UserWarning code/eval_typing_2layer.py   --do_eval   --do_lower_case   --data_dir ../../data/data/OpenEntity   --ernie_model $model_base   --max_seq_length 256   --train_batch_size 16   --learning_rate 2e-5   --num_train_epochs 10.0   --output_dir output_open_2layer --threshold 0.3 --fp16 --loss_scale 128 --K_V_dim 100 --Q_dim 768 --self_att --data_token open
