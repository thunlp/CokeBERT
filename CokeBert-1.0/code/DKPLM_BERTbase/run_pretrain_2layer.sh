#BEST
rm pretrain_out_2layer/*
#Q=Q_linear(Q) --> bais | activation
#K=K_V_linear(K) --> No bais | no activation
#V=V_linear(V) --> No bais | no activation

#K = self.K_V_linear
#V=self.V_linear

#normalize(hop1,hop2), CLS query (hop1,hop2), org transE, emb 200 dim, bias=False

CUDA_VISIBLE_DEVICES=0,1,2 python3 -W ignore::UserWarning code/run_pretrain_2layer.py --do_train --data_dir ../../data/pretrain_data_bert/merge --bert_model ../../data/bert_base --output_dir pretrain_out_2layer/ --task_name pretrain --fp16 --max_seq_length 256 --K_V_dim 100 --Q_dim 768 --train_batch_size 32 --self_att #--graphsage
#train_batch_size 32






