from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
import pickle
import random
import logging
import argparse
import datetime
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW

import coke.iterators as iterators
import coke.indexed_dataset as indexed_dataset
from coke import CokeBertForPreTraining
from coke.training_args import coke_training_args
from coke.utils import (
    load_ent_emb_dynamic,
    load_ent_emb_static,
    k_v,
    load_k_v_queryR_small,
    load_kg_embedding
)



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def main():

    args = coke_training_args()

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

    args.output_dir = os.path.join(args.output_dir, 'pretrain_coke-' + args.backbone + f'-{args.neighbor_hop}')
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare model
    model_name = 'coke-' + args.backbone
    model = CokeBertForPreTraining.from_pretrained(os.path.join('../checkpoint', model_name), neighbor_hop=args.neighbor_hop)
    model.to(device)

    def collate_fn(x):
        x = torch.LongTensor(np.array(x))
        entity_idx = x[:, 4*args.max_seq_length:5*args.max_seq_length]
        uniq_idx = np.unique(entity_idx.numpy())
        ent_candidate = torch.LongTensor(uniq_idx+1) #del
        ent_candidate = ent_candidate.repeat([torch.cuda.device_count(), 1]) #batch

        # build entity labels
        d = {}
        dd = []
        for i, idx in enumerate(uniq_idx):
            d[idx] = i
            dd.append(idx)
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
        ent_labels = entity_idx.clone()
        d[-1] = -1
        ent_labels = ent_labels.apply_(lambda x: d[x])

        entity_idx.apply_(map)

        ent_emb = entity_idx + 1
        mask = entity_idx.clone()
        mask.apply_(lambda x: 0 if x == -1 else 1)
        mask[:,0] = 1

        return (
            x[:,:args.max_seq_length], 
            x[:,args.max_seq_length:2*args.max_seq_length], 
            x[:,2*args.max_seq_length:3*args.max_seq_length], 
            x[:,3*args.max_seq_length:4*args.max_seq_length], 
            ent_emb, 
            mask, 
            x[:,6*args.max_seq_length:], 
            ent_candidate, 
            ent_labels
        )


    global_step = 0
    if args.do_train:

        # Prepare KG data
        logger.info('Loading KG ...')
        ent_neighbor, ent_r, ent_outORin = load_ent_emb_static(args.data_dir)
        embed_ent, embed_r = load_kg_embedding(args.data_dir)
        logger.info('Finish loading')
        
        # Prepare text dataset
        train_data = indexed_dataset.IndexedDataset(os.path.join(args.data_dir, args.backbone), fix_lua_indexing=True)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_sampler = BatchSampler(train_sampler, args.train_batch_size, True)
        train_iterator = iterators.EpochBatchIterator(train_data, collate_fn, train_sampler)
        
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
        num_train_steps = int(
            len(train_data) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        model.train()

        fout = open(os.path.join(args.output_dir, "loss.{}".format(datetime.datetime.now())), 'w')
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_iterator.next_epoch_itr(), desc="Iteration")):

                batch = tuple(t.to(device) for t in batch)
                
                input_ids, input_mask, token_type_ids, mlm_labels, input_ent, ent_mask, nsp_labels, ent_candidate, ent_labels = batch
                input_ent += 1
                k_1, v_1, k_2, v_2, k_cand_1, v_cand_1, k_cand_2, v_cand_2, cand_pos_tensor = load_k_v_queryR_small(input_ent, ent_candidate, ent_neighbor, ent_r, ent_outORin, embed_ent, embed_r, args.neighbor_hop)
                cand_pos_tensor = cand_pos_tensor.to(device)

                loss, original_loss = model(
                    input_ids, input_mask, token_type_ids, labels=mlm_labels, input_ent=input_ent, ent_mask=ent_mask, next_sentence_label=nsp_labels,
                    candidate=ent_candidate, ent_labels=ent_labels, k_1=k_1, v_1=v_1, k_2=k_2, v_2=v_2, 
                    k_cand_1=k_cand_1, v_cand_1=v_cand_1, k_cand_2=k_cand_2, v_cand_2=v_cand_2, cand_pos_tensor=cand_pos_tensor)


                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                end_time_4 = time.time()

                fout.write("{} {}\n".format(loss.item()*args.gradient_accumulation_steps, original_loss.item()))
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
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


if __name__ == "__main__":
    main()

