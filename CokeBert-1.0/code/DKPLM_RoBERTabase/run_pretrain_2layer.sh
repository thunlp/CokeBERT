#BEST
rm pretrain_out_2layer/*
#Q=Q_linear(Q) --> bais | activation
#K=K_V_linear(K) --> No bais | no activation
#V=V_linear(V) --> No bais | no activation

#K = self.K_V_linear
#V=self.V_linear

#normalize(hop1,hop2), CLS query (hop1,hop2), org transE, emb 200 dim, bias=False

#model_base=ernie_roberta_base
model_base=../../data/roberta_base
#model_base=ernie_base

CUDA_VISIBLE_DEVICES=5,6,7 python3 -W ignore::UserWarning code/run_pretrain_2layer.py --do_train --data_dir ../../data/pretrain_data_roberta/merge --bert_model $model_base --output_dir pretrain_out_2layer/ --task_name pretrain --fp16 --max_seq_length 256 --K_V_dim 100 --Q_dim 768 --train_batch_size 32 --self_att #--graphsage
#train_batch_size 32






