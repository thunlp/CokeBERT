rm output_tacred_2layer/*

model=../../data/DKPLM_BERTbase_2layer

# train
#max_seq_length 256->128
# threshold 0.4 --> 6.0
CUDA_VISIBLE_DEVICES=2,6 python3 -W ignore::UserWarning code/run_tacred_2layer.py   --do_train   --do_lower_case   --data_dir ../../data/data/tacred   --ernie_model $model   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 6.0   --output_dir output_tacred_2layer   --fp16   --loss_scale 128 --threshold 0.4 --K_V_dim 100 --Q_dim 768 --self_att #--graphsage

# evaluate
#rm output_tacred_new_n_CLS_comb200/*comb


#max_seq_length 256->128
# threshold 0.4 --> 6.0
CUDA_VISIBLE_DEVICES=2,6 python3 -W ignore::UserWarning code/eval_tacred_2layer.py   --do_eval   --do_lower_case   --data_dir ../../data/data/tacred   --ernie_model $model   --max_seq_length 256   --train_batch_size 32   --learning_rate 2e-5   --num_train_epochs 6.0   --output_dir output_tacred_2layer   --fp16   --loss_scale 128 --threshold 0.4 --K_V_dim 100 --Q_dim 768 --self_att #--graphsage

python3 code/score.py output_tacred_2layer
