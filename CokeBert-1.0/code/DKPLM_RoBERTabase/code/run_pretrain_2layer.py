# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import math
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from knowledge_bert.tokenization_roberta import RobertaTokenizer
#from knowledge_bert.tokenization import BertTokenizer
from knowledge_bert.modeling_new_n_CLS_comb200_rob import BertForPreTraining
from knowledge_bert.optimization import BertAdam
from knowledge_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
####add###
from random import randint
##########
import pickle
import time

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0


def load_ent_emb_static():
    with open('../../data/load_data_n/e1_e2_list_2D_Tensor.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data_test/e1_e2_list_2D_Tensor.pkl', 'rb') as f:
        ent_neighbor = pickle.load(f)
    print("ent_neighbor:",ent_neighbor.shape)

    with open('../../data/load_data_n/e1_r_list_2D_Tensor.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data_test/e1_r_list_2D_Tensor.pkl', 'rb') as f:
        ent_r = pickle.load(f)
    print("ent_r:",ent_r.shape)

    with open('../../data/load_data_n/e1_outORin_list_2D_Tensor.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data_test/e1_outORin_list_2D_Tensor.pkl', 'rb') as f:
        ent_outORin = pickle.load(f)
    print("ent_outORin:",ent_outORin.shape)

    #ent_neighbor = torch.nn.Embedding.from_pretrained(ent_neighbor)
    #ent_r = torch.nn.Embedding.from_pretrained(ent_r)
    #ent_outORin = torch.nn.Embedding.from_pretrained(ent_outORin)

    return ent_neighbor, ent_r, ent_outORin


def load_knowledge():
    #load KG emb
    vecs = []
    vecs.append([0]*100) # CLS
    with open("../../data/kg_embed/entity2vec.vec", 'r') as fin:
    #with open("kg_embed/entity2vec.del", 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed_ent = torch.FloatTensor(vecs)
    #embed = torch.nn.Embedding.from_pretrained(embed)
    #logger.info("Shape of entity embedding: "+str(embed.weight.size()))
    del vecs


    #load relation emb
    vecs = []
    vecs.append([0]*100) # CLS
    with open("../../data/kg_embed/relation2vec.vec", 'r') as fin:
    #with open("kg_embed/relation2vec.del", 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed_r = torch.FloatTensor(vecs)
    #embed = torch.nn.Embedding.from_pretrained(embed)
    #logger.info("Shape of entity embedding: "+str(embed.weight.size()))
    del vecs


    #embed_ent = torch.nn.Embedding.from_pretrained(embed_ent)
    #embed_r = torch.nn.Embedding.from_pretrained(embed_r)

    return embed_ent, embed_r



def load_k_v_queryR_small(input_ent, candidate):
        #print(input_ent)
        #print(input_ent.shape)
        #exit()
        input_ent = input_ent.cpu()
        candidate = candidate.cpu()


        #print(input_ent)
        #print(input_ent.shape)
        #print("======")

        ent_pos_s = torch.nonzero(input_ent)
        #print(ent_pos_s)
        #print(ent_pos_s.shape)
        #print("======")

        max_entity=0
        value=0
        idx_1 = 0
        last_part = 0
        for idx_2,x in enumerate(ent_pos_s):
            if int(x[0]) != value:
                max_entity = max(idx_2-idx_1,max_entity)
                idx_1 = idx_2
                value = int(x[0])
                last_part = 1
            else:
                last_part+=1
        max_entity = max(last_part,max_entity)
        #print("\n")
        #print(max_entity)
        #print("======")

        #ents = input_ent[input_ent!=0]

        new_input_ent = list()
        #print(input_ent.shape) #(batch_size,256)
        cand_pos_tensor = torch.LongTensor(input_ent.shape[0],candidate.shape[1]) #V
        #(batch_size,uniq_entity)
        for i_th, ten in enumerate(input_ent):
            ten_ent = ten[ten!=0]
            #a=time.time()
            ten_ent_uniqe = ten_ent.unique() #v
            #print(ten_ent_uniqe)
            #print(ten_ent_uniqe.shape)
            #print(candidate[0])
            #print(candidate[0].shape)
            cand_pos_tensor[i_th][:] = torch.LongTensor([int(ent in ten_ent_uniqe) for ent in candidate[0]]) #V
            #b=time.time()
            #print("\n")
            #print(b-a)
            #exit()

            new_input_ent.append( torch.cat( (ten_ent,( torch.LongTensor( [0]*(max_entity-ten_ent.shape[0]) ) ) ) ) )
        input_ent = torch.stack(new_input_ent)

        #print(cand_pos_tensor)
        #print(cand_pos_tensor.shape)

        def k_v(input_ent):

            #print(input_ent)
            #print(input_ent.shape)

            #Neighbor
            input_ent_neighbor = torch.index_select(ent_neighbor,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1])).long()

            #create input_ent_neighbor_1
            input_ent_neighbor_emb_1 = torch.index_select(embed_ent,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1])) #
            #input_ent_neighbor_emb_1 = input_ent_neighbor_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],ent_neighbor.shape[1],100)
            input_ent_neighbor_emb_1 = input_ent_neighbor_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],ent_neighbor.shape[1],embed_ent.shape[-1])

            #create input_ent_r_1:
            input_ent_r_emb_1 = torch.index_select(ent_r,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1])).long()
            input_ent_r_emb_1 = torch.index_select(embed_r,0,input_ent_r_emb_1.reshape(input_ent_r_emb_1.shape[0]*input_ent_r_emb_1.shape[1])) #
            #input_ent_r_emb_1 = input_ent_r_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],ent_r.shape[1],100)
            input_ent_r_emb_1 = input_ent_r_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],ent_r.shape[1],embed_r.shape[-1])

            #create outORin_1:
            input_ent_outORin_emb_1 = torch.index_select(ent_outORin,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1]))
            input_ent_outORin_emb_1 = input_ent_outORin_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],input_ent_outORin_emb_1.shape[1])
            input_ent_outORin_emb_1 = input_ent_outORin_emb_1.unsqueeze(3)

            #create input_ent_neighbor_2
            input_ent_neighbor_2 = torch.index_select(ent_neighbor,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1])).long()
            input_ent_neighbor_emb_2 = torch.index_select(embed_ent,0,input_ent_neighbor_2.reshape(input_ent_neighbor_2.shape[0]*input_ent_neighbor_2.shape[1])) #
            input_ent_neighbor_emb_2 = input_ent_neighbor_emb_2.reshape(input_ent.shape[0],input_ent.shape[1],ent_neighbor.shape[1],ent_neighbor.shape[1],100)

            #create input_ent_r_2:
            input_ent_r_2 = torch.index_select(ent_r,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1])).long()
            input_ent_r_emb_2 = torch.index_select(embed_r,0,input_ent_r_2.reshape(input_ent_r_2.shape[0]*input_ent_r_2.shape[1])) #
            input_ent_r_emb_2 = input_ent_r_emb_2.reshape(input_ent.shape[0],input_ent.shape[1],ent_r.shape[1],ent_neighbor.shape[1],100)

            #create outORin_2:
            input_ent_outORin_emb_2 = torch.index_select(ent_outORin,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1]))
            input_ent_outORin_emb_2 = input_ent_outORin_emb_2.reshape(input_ent_r_emb_2.shape[0],input_ent_r_emb_2.shape[1],input_ent_r_emb_2.shape[2],input_ent_r_emb_2.shape[3])
            input_ent_outORin_emb_2 = input_ent_outORin_emb_2.unsqueeze(4)

            ###
            k_1 = input_ent_outORin_emb_1.cuda()*input_ent_r_emb_1.cuda()
            v_1 = input_ent_neighbor_emb_1.cuda()+k_1
            k_2 = input_ent_outORin_emb_2.cuda()*input_ent_r_emb_2.cuda()
            v_2 = input_ent_neighbor_emb_2.cuda()+k_2

            return k_1,v_1,k_2,v_2


        #print(input_ent)
        #print(input_ent.shape)
        k_1,v_1,k_2,v_2 = k_v(input_ent)

        #input_cand = candidate[0] * cand_pos_tensor
        #print(input_cand)
        #print(input_cand.shape)
        #exit()
        k_cand_1,v_cand_1,k_cand_2,v_cand_2 = k_v(candidate)
        #print(v_cand)
        #print(v_cand.shape)
        #exit()
        return k_1,v_1,k_2,v_2,k_cand_1,v_cand_1,k_cand_2,v_cand_2,cand_pos_tensor.cuda()




print("Load Emb ...")
embed_ent, embed_r = load_knowledge()
#embed_ent = torch.nn.Embedding.from_pretrained(embed_ent)
#embed_ent = torch.nn.Embedding.from_pretrained(embed_ent).requires_grad_(False)
#embed_ent.weight.requires_grad=False
#embed_r = torch.nn.Embedding.from_pretrained(embed_r)
#embed_r = torch.nn.Embedding.from_pretrained(embed_r).requires_grad_(False)
#embed_r.weight.requires_grad=False

ent_neighbor, ent_r, ent_outORin = load_ent_emb_static()
#ent_neighbor = torch.nn.Embedding.from_pretrained(ent_neighbor)
#ent_neighbor = torch.nn.Embedding.from_pretrained(ent_neighbor).requires_grad_(False)
#ent_neighbor.weight.requires_grad=False
#ent_r = torch.nn.Embedding.from_pretrained(ent_r)
#ent_r = torch.nn.Embedding.from_pretrained(ent_r).requires_grad_(False)
#ent_r.weight.requires_grad=False
#ent_outORin = torch.nn.Embedding.from_pretrained(ent_outORin)
#ent_outORin = torch.nn.Embedding.from_pretrained(ent_outORin).requires_grad_(False)
#ent_outORin.weight.requires_grad=False

#ent_neighbor, ent_r, ent_outORin = load_ent_emb_dynamic()
print("Finsh loading Emb")



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        #default=3.0,
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    ##########ADD##
    parser.add_argument("--K_V_dim",
                        type=int,
                        default=100,
                        help="Key and Value dim == KG representation dim")

    parser.add_argument("--Q_dim",
                        type=int,
                        default=768,
                        help="Query dim == Bert six output layer representation dim")
    parser.add_argument('--graphsage',
                        default=False,
                        action='store_true',
                        help="Whether to use Attention GraphSage instead of GAT")
    parser.add_argument('--self_att',
                        default=True,
                        action='store_true',
                        help="Whether to use GAT")
    ###############

    args = parser.parse_args()


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()


    train_data = None
    num_train_steps = None
    if args.do_train:
        # TODO
        import indexed_dataset
        from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler
        import iterators
        #train_data = indexed_dataset.IndexedCachedDataset(args.data_dir)
        train_data = indexed_dataset.IndexedDataset(args.data_dir, fix_lua_indexing=True)
        #print(train_data)
        #print("-----------")
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_sampler = BatchSampler(train_sampler, args.train_batch_size, True)
        def collate_fn(x):
            x = torch.LongTensor([xx for xx in x])
            #x = torch.LongTensor([xx%9 for xx in x]) ##

            ###
            #entity_idx = x[:, 4*args.max_seq_length:5*args.max_seq_length]
            ###
            ###
            entity_idx = x[:, 3*args.max_seq_length:4*args.max_seq_length]
            ###
            #entity_idx = x[:, 4*args.max_seq_length]
            #print(entity_idx)
            #print(entity_idx.shape)

            # Build candidate
            #
            #print(entity_idx)
            uniq_idx = np.unique(entity_idx.numpy())
            #print(uniq_idx)
            #print(uniq_idx.shape)
            #exit()
            #ent_candidate = embed(torch.LongTensor(uniq_idx+1))
            #print(ent_candidate)
            #print(ent_candidate.shape)
            ent_candidate = torch.LongTensor(uniq_idx+1) #del
            #print(ent_candidate)
            #print(ent_candidate.shape)
            #print(ent_candidate)
            #print(ent_candidate.shape)
            #print(ent_candidate)
            #print(ent_candidate.shape)
            #print("================")
            ent_candidate = ent_candidate.repeat([n_gpu, 1]) #batch
            #print(ent_candidate)
            #print(ent_candidate.shape)
            #exit()
            #print(ent_candidate)
            #print(ent_candidate.shape)
            #exit()
            #ent_candidate = embed(torch.LongTensor(uniq_idx+1)) #del
            #print(ent_candidate)
            #print(ent_candidate.shape)
            #print(ent_candidate.size())
            #exit()
            #
            #ent_candidate = torch.LongTensor(ent_candidate+1) #single
            #! --> return uniq_idx =>all entity in batch

            # build entity labels
            d = {}
            dd = []
            for i, idx in enumerate(uniq_idx):
                d[idx] = i
                dd.append(idx)
            ###
            '''
            ent_size = len(uniq_idx)-1
            def map(x):
                if x == -1:
                    return -1
                else:
                    rnd = random.uniform(0, 1)
                    if rnd < 0.05:
                        return dd[random.randint(1, ent_size)]
                    elif rnd < 0.2:
                        return -1
                    else:
                        return x
            '''
            ###
            ent_labels = entity_idx.clone()
            d[-1] = -1
            ent_labels = ent_labels.apply_(lambda x: d[x])

            ###
            '''
            entity_idx.apply_(map)

            #ent_emb = embed(entity_idx+1)
            ent_emb = entity_idx+1 ##
            #! --> return entity+1 => input_ent

            mask = entity_idx.clone()
            mask.apply_(lambda x: 0 if x == -1 else 1)
            mask[:,0] = 1
            '''
            ###

            ###
            # entity_idx.apply_(map)
            #mask = entity_idx.clone()
            #mask.apply_(lambda x: 0 if x == -1 else 1)
            mask = x[:, 4*args.max_seq_length:5*args.max_seq_length]
            mask[:,0] = 1

            entity_idx = entity_idx * mask
            ###
            entity_idx[entity_idx == 0] = -1
            ###
            ent_emb = entity_idx+1
            ###

            ###
            #return x[:,:args.max_seq_length], x[:,args.max_seq_length:2*args.max_seq_length], x[:,2*args.max_seq_length:3*args.max_seq_length], x[:,3*args.max_seq_length:4*args.max_seq_length], ent_emb, mask, x[:,6*args.max_seq_length:], ent_candidate, ent_labels
            ###
            ###
            return x[:,:args.max_seq_length], x[:,args.max_seq_length:2*args.max_seq_length], x[:,2*args.max_seq_length:3*args.max_seq_length], ent_emb, mask, x[:,5*args.max_seq_length:], ent_candidate, ent_labels
            ###

        train_iterator = iterators.EpochBatchIterator(train_data, collate_fn, train_sampler)
        num_train_steps = int(
            len(train_data) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    #model, missing_keys = BertForPreTraining.from_pretrained(args.bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
    model, missing_keys = BertForPreTraining.from_pretrained(args.bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank), args=args)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_linear = ['layer.2.output.dense_ent', 'layer.2.intermediate.dense_1', 'bert.encoder.layer.2.intermediate.dense_1_ent', 'layer.2.output.LayerNorm_ent']
    no_linear = [x.replace('2', '11') for x in no_linear]
    param_optimizer = [(n, p) for n, p in param_optimizer if not any(nl in n for nl in no_linear)]
    #param_optimizer = [(n, p) for n, p in param_optimizer if not any(nl in n for nl in missing_keys)]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'LayerNorm_ent.bias', 'LayerNorm_ent.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        #logger.info(dir(optimizer))
        #op_path = os.path.join(args.bert_model, "pytorch_op.bin")
        #optimizer.load_state_dict(torch.load(op_path))

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    global_step = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        model.train()
        import datetime
        fout = open(os.path.join(args.output_dir, "loss.{}".format(datetime.datetime.now())), 'w')
        more_than_one_2 = 0
        less_than_one_2 = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            ###
            '''
            for step, batch in enumerate(tqdm(train_iterator.next_epoch_itr(), desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, masked_lm_labels, input_ent, ent_mask, next_sentence_label, ent_candidate, ent_labels = batch
            '''
            ###

            ###
            if args.local_rank == 0 or args.local_rank == -1:
                iters = tqdm(train_iterator.next_epoch_itr(), desc="Iteration")
            else:
                iters = train_iterator.next_epoch_itr()

            for step, batch in enumerate(iters):
                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, masked_lm_labels, input_ent, ent_mask, next_sentence_label, ent_candidate, ent_labels = batch
                ###
                ###
                #print(len(input_ids[input_ids==2]))
                if len(input_ids[input_ids==2]) != args.train_batch_size:
                    for i_th_1, input_id in enumerate(input_ids):
                        print(input_id[input_id==2])
                        print(len(input_id[input_id==2]))
                        if len(input_id[input_id==2]) > 1:
                            for i_th_2 ,id in enumerate(input_id):
                                if id == 2:
                                    print("Befor:",input_id)
                                    input_ids[i_th_1][i_th_2] = 0
                                    more_than_one_2 += 1
                                    print("more_than_one_2:",more_than_one_2)
                                    print("After:",input_id)
                                    if len(input_id[input_id==2] == 1):
                                        break
                        elif len(input_id[input_id==2]) < 1:
                            print("Error!! Have no id=2 </s>")
                            less_than_one_2 += 1
                            print("less_than_one_2:",less_than_one_2)
                            print(input_id)
                            input_ids[i_th_1][-1] = 2
                        else:
                            print("ids_2 == 1")
                ###
                ###

                #start_time_1 = time.time()
                k_1, v_1, k_2, v_2, k_cand_1, v_cand_1, k_cand_2, v_cand_2, cand_pos_tensor = load_k_v_queryR_small(input_ent,ent_candidate)
                #k, v = load_k_v_queryR(input_ent,device)
                #input_ent_neighbor_emb, input_ent_r_emb, input_ent_outORin_emb = load_k_v_queryR(input_ent)
                #end_time_1 = time.time()
                #print("load_k_v_queryR:{}".format(end_time_1-start_time_1))
                #print(ent_candidate)
                #print(ent_candidate.shape)
                #exit()
                #k_cand, v_cand = load_k_v_queryR_small(ent_candidate,"candidate")
                #k_cand, v_cand = load_k_v_queryR(ent_candidate,device)
                #input_ent_neighbor_emb_c, input_ent_r_emb_c, input_ent_outORin_emb_c = load_k_v_queryR(candidate)
                #end_time_2 = time.time()
                #print("load_cand:{}".format(end_time_2-end_time_1))

                #k, v = load_batch_k_v_queryE(input_ent,500)
                #k_cand, v_cand = load_batch_k_v_queryE(ent_candidate,500)
                #k, v = load_batch_k_v_queryR(input_ent,300)
                #k_cand, v_cand = load_batch_k_v_queryR(ent_candidate,300)

                if args.fp16:
                    #loss, original_loss = model(input_ids, segment_ids, input_mask, masked_lm_labels, input_ent, ent_mask, next_sentence_label, ent_candidate, ent_labels, k_1.half(), v_1.half(), k_2.half(), v_2.half(),  k_cand_1.half(), v_cand_1.half(), k_cand_2.half(), v_cand_2.half(), cand_pos_tensor)
                    loss, original_loss = model(input_ids, None, input_mask, masked_lm_labels, input_ent, ent_mask, next_sentence_label, ent_candidate, ent_labels, k_1.half(), v_1.half(), k_2.half(), v_2.half(),  k_cand_1.half(), v_cand_1.half(), k_cand_2.half(), v_cand_2.half(), cand_pos_tensor)
                else:
                    #loss, original_loss = model(input_ids, segment_ids, input_mask, masked_lm_labels, input_ent, ent_mask, next_sentence_label, ent_candidate, ent_labels, k_1, v_1, k_2, v_2, k_cand_1, v_cand_1, k_cand_2, v_cand_2, cand_pos_tensor)
                    loss, original_loss = model(input_ids, None, input_mask, masked_lm_labels, input_ent, ent_mask, next_sentence_label, ent_candidate, ent_labels, k_1, v_1, k_2, v_2, k_cand_1, v_cand_1, k_cand_2, v_cand_2, cand_pos_tensor)


                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                    original_loss = original_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                end_time_4 = time.time()
                #print("bp time:{}".format(end_time_4))
                #print("=====================================")

                fout.write("{} {}\n".format(loss.item()*args.gradient_accumulation_steps, original_loss.item()))
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % 100000 == 0:
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin_{}".format(global_step))
                        torch.save(model_to_save.state_dict(), output_model_file)

        fout.close()

    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

    # Save the optimizer
    #output_optimizer_file = os.path.join(args.output_dir, "pytorch_op.bin")
    #torch.save(optimizer.state_dict(), output_optimizer_file)

    # Load a trained model that you have fine-tuned
    # model_state_dict = torch.load(output_model_file)
    # model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict)
    # model.to(device)

    # if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     eval_examples = processor.get_dev_examples(args.data_dir)
    #     eval_features = convert_examples_to_features(
    #         eval_examples, label_list, args.max_seq_length, tokenizer)
    #     logger.info("***** Running evaluation *****")
    #     logger.info("  Num examples = %d", len(eval_examples))
    #     logger.info("  Batch size = %d", args.eval_batch_size)
    #     all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    #     all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    #     all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    #     all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    #     eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    #     # Run prediction for full data
    #     eval_sampler = SequentialSampler(eval_data)
    #     eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    #     model.eval()
    #     eval_loss, eval_accuracy = 0, 0
    #     nb_eval_steps, nb_eval_examples = 0, 0
    #     for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
    #         input_ids = input_ids.to(device)
    #         input_mask = input_mask.to(device)
    #         segment_ids = segment_ids.to(device)
    #         label_ids = label_ids.to(device)

    #         with torch.no_grad():
    #             tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
    #             logits = model(input_ids, segment_ids, input_mask)

    #         logits = logits.detach().cpu().numpy()
    #         label_ids = label_ids.to('cpu').numpy()
    #         tmp_eval_accuracy = accuracy(logits, label_ids)

    #         eval_loss += tmp_eval_loss.mean().item()
    #         eval_accuracy += tmp_eval_accuracy

    #         nb_eval_examples += input_ids.size(0)
    #         nb_eval_steps += 1

    #     eval_loss = eval_loss / nb_eval_steps
    #     eval_accuracy = eval_accuracy / nb_eval_examples

    #     result = {'eval_loss': eval_loss,
    #               'eval_accuracy': eval_accuracy,
    #               'global_step': global_step,
    #               'loss': tr_loss/nb_tr_steps}

    #     output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    #     with open(output_eval_file, "w") as writer:
    #         logger.info("***** Eval results *****")
    #         for key in sorted(result.keys()):
    #             logger.info("  %s = %s", key, str(result[key]))
    #             writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()

