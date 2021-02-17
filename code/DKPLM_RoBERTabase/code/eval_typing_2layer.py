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
import simplejson as json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

###
from knowledge_bert.tokenization_roberta import RobertaTokenizer
from knowledge_bert.typing_rob import RobertaTokenizer as RobertaTokenizer_label
from knowledge_bert.modeling_new_n_CLS_comb200_rob import BertForEntityTyping
###

from knowledge_bert.optimization import BertAdam
from knowledge_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import time
import pickle
import random


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, input_ent, ent_mask, labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labels = labels
        self.input_ent = input_ent
        self.ent_mask = ent_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r") as f:
            return json.load(f)


class TypingProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.json")))
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")
        d = {}
        for e in examples:
            for l in e.label:
                if l in d:
                    d[l] += 1
                else:
                    d[l] = 1
        for k, v in d.items():
            d[k] = (len(examples) - v) * 1. /v
        return examples, list(d.keys()), d

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")


    def get_label(self, data_dir):
        """See base class."""
        return ["0", "1"]
        '''
        filename = os.path.join(data_dir, "label.dict")
        v = []
        with open(filename, "r") as fin:
            for line in fin:
                vec = line.split("\t")
                v.append(vec[1])
        return v
        '''
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text_a = (line['sent'], [["SPAN", line["start"], line["end"]]])
            text_b = line['ents']
            label = line['labels']
            #if guid != 51:
            #    continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



def load_ent_emb_static():

    with open('../../data/load_data_n/e1_e2_list_2D_Tensor.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data/e1_e2_list_2D_Tensor.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data_test/e1_e2_list_2D_Tensor.pkl', 'rb') as f:
        ent_neighbor = pickle.load(f)

    with open('../../data/load_data_n/e1_r_list_2D_Tensor.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data/e1_r_list_2D_Tensor.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data_test/e1_r_list_2D_Tensor.pkl', 'rb') as f:
        ent_r = pickle.load(f)

    with open('../../data/load_data_n/e1_outORin_list_2D_Tensor.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data/e1_outORin_list_2D_Tensor.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data_test/e1_outORin_list_2D_Tensor.pkl', 'rb') as f:
        ent_outORin = pickle.load(f)

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


def load_k_v_queryR_small(input_ent):
        #print(input_ent)
        #print(input_ent.shape)
        #exit()
        input_ent = input_ent.cpu()

        #if input_ent.shape[0]>1:
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
        for i_th, ten in enumerate(input_ent):
            ten_ent = ten[ten!=0]
            new_input_ent.append( torch.cat( (ten_ent,( torch.LongTensor( [0]*(max_entity-ten_ent.shape[0]) ) ) ) ) )
            #print(new_input_ent[i_th].shape)
            #print("---------------")
            #print(torch.nonzero(ten))
            #non_ = torch.nonzero(ten)
            #print(torch.nonzero(input_ent[i_th]),
            #torch.LongTensor([0]*max_entity-input_ent[i_th]))
        #print(new_input_ent)
        #print("======")
        input_ent = torch.stack(new_input_ent)
        #print(input_ent)
        #print(input_ent.shape)
        #exit()


        #Neighbor
        input_ent_neighbor = torch.index_select(ent_neighbor,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1])).long()

        #create input_ent_neighbor_1
        input_ent_neighbor_emb_1 = torch.index_select(embed_ent,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1])) #
        input_ent_neighbor_emb_1 = input_ent_neighbor_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],ent_neighbor.shape[1],100)

        #create input_ent_r_1:
        input_ent_r_emb_1 = torch.index_select(ent_r,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1])).long()
        input_ent_r_emb_1 = torch.index_select(embed_r,0,input_ent_r_emb_1.reshape(input_ent_r_emb_1.shape[0]*input_ent_r_emb_1.shape[1])) #
        input_ent_r_emb_1 = input_ent_r_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],ent_r.shape[1],100)

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
        #exit()
        return k_1,v_1,k_2,v_2



print("Load Emb ...")
embed_ent, embed_r = load_knowledge()
ent_neighbor, ent_r, ent_outORin = load_ent_emb_static()
#ent_neighbor, ent_r, ent_outORin = load_ent_emb_dynamic()
print("Finsh loading Emb")


def check_pre(a, b):
    if len(a) < len(b):
        return False
    else:
        #a = [x for x in a if x != 'ĠUCHIJ' and x != 'ĠTG' and x != 'ĠUKIP' and x != 'ĠCLSID']
        a = [x for x in a if x != 'ĠUCHIJ' and x != 'ĠCLSID']
        for i in range(len(b)):
            if a[i] != b[i]:
                return False
        return True



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer_label, tokenizer, threshold):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label : i for i, label in enumerate(label_list)}

    entity2id = {}
    with open("../../data/kg_embed/entity2id.txt") as fin:
        fin.readline()
        for line in fin:
            qid, eid = line.strip().split('\t')
            entity2id[qid] = int(eid)

    features = []
    for (ex_index, example) in enumerate(examples):
        ex_text_a = example.text_a[0]
        h = example.text_a[1][0]

        ###
        #n
        if 'CLSID ' in ex_text_a:
            print("=======")
            print("CLSID in the setence:")
            print(ex_text_a)
            print("=======")
            exit()
        if ' UCHIJ' in ex_text_a:
            print("=======")
            print("UCHIJ in the setence:")
            print(ex_text_a)
            print("=======")
            exit()
        #assert 'UCHIJ' not in ex_text_a
        ###

        #ex_text_a = ex_text_a[:h[1]] + "。 " + ex_text_a[h[1]:h[2]] + " 。" + ex_text_a[h[2]:]

        ###
        #n
        ex_text_a = ex_text_a[:h[1]] + "CLSID " + ex_text_a[h[1]:h[2]] + " UCHIJ" + ex_text_a[h[2]:]
        ###

        '''
        begin, end = h[1:3]
        ### CLS and #TJ: begining of the entity and the end of end (entity range)
        #h[1] += 2
        #h[2] += 2
        h[1] += 6
        h[2] += 6
        '''


        first_token = ex_text_a.split(" ")[0]
        if first_token == "CLSID":
            tokens_a = tokenizer.tokenize(" "+ex_text_a)
            #tokens_a = tokenizer.tokenize(ex_text_a)
        else:
            #tokens_a = tokenizer.tokenize(ex_text_a)
            tokens_a = tokenizer.tokenize(" "+ex_text_a)


        ####
        #new
        CLSID_list = list()
        GCLSID_list = list()
        UCHIJ_list = list()
        GUCHIJ_list = list()
        entities_a = list()
        typing_flag = False
        entities_a = ["UNK"]*len(tokens_a)
        for i_th, token in enumerate(tokens_a):
            if 'CLSID' == token:
                CLSID_list.append(token)
            if 'ĠCLSID' == token:
                GCLSID_list.append(token)
            if 'UCHIJ' == token:
                UCHIJ_list.append(token)
            if 'ĠUCHIJ' == token:
                GUCHIJ_list.append(token)

            if typing_flag==True:
                entities_a[i_th] = h[0]

            if token =='CLSID' or token == 'ĠCLSID':
            #if token == 'ĠCLSID':
                typing_flag=True
            if token == 'UCHIJ' or token == 'ĠUCHIJ':
            #if token == 'ĠUCHIJ':
                typing_flag=False
                entities_a[i_th] = "UNK"

        if len(CLSID_list)+len(GCLSID_list) != 1:
            print("=======")
            print("CLSID Wrong")
            print("=======")
            print(ex_text_a)
            print("---")
            print(tokens_a)
            print("---")
            print(entities_a)
            ################
            ################
            entities_a = ["UNK"]*len(tokens_a)
            CLSID_id_list=list()
            UCHIJ_id_list=list()
            for i_th, token in enumerate(tokens_a):
                if token == 'ĠCLSID':
                    CLSID_id_list.append(i_th)
                if token == 'ĠUCHIJ':
                    UCHIJ_id_list.append(i_th)
                    break

            CLSID_id = CLSID_id_list[-1]+1
            UCHIJ_id = UCHIJ_id_list[-1]
            entities_a[CLSID_id:UCHIJ_id] = h[0]*(UCHIJ_id-CLSID_id)
            for CLSID_id in CLSID_id_list[:-1]:
                tokens_a[CLSID_id] = 'CLSID'

        if len(UCHIJ_list)+len(GUCHIJ_list) != 1:
            print("=======")
            print("UCHIJ Wrong")
            print("=======")
            print(ex_text_a)
            print("---")
            print(tokens_a)
            print("---")
            print(entities_a)
            ################
            ################
            #Maybe wrong !! check
            entities_a = ["UNK"]*len(tokens_a)
            CLSID_id_list=list()
            UCHIJ_id_list=list()
            for i_th, token in enumerate(tokens_a):
                if token == 'ĠCLSID':
                    CLSID_id_list.append(i_th)
                if token == 'ĠUCHIJ':
                    UCHIJ_id_list.append(i_th)
                    break
            CLSID_id = CLSID_id_list[-1]+1
            UCHIJ_id = UCHIJ_id_list[-1]
            entities_a[CLSID_id:UCHIJ_id] = h[0]*(UCHIJ_id-CLSID_id)
            for UCHIJ_id in UCHIJ_id_list[:-1]:
                tokens_a[UCHIJ_id] = 'UCHIJ'
        ####





        # change begin pos
        ent_pos = [x for x in example.text_b if x[-1]>threshold]
        ###
        for x in ent_pos:
            '''
            if x[1] > end: #CLS, #CLSID, #UCHIJ ??
                #x[1] += 4
                x[1] += 12
                x[2] += 12
            elif x[1] >= begin: #CLS, #CLSID
                #x[1] += 2
                x[1] += 6
                x[2] += 6
            '''
            x[-1] = example.text_a[0][x[1]:x[2]]
        ##

        ####
        #new
        ######
        #tokens_a = tokenizer.tokenize(" " + ex_text_a)
        #entities_a = ["UNK"] * len(tokens_a)
        entities = ["UNK"] * len(tokens_a)
        for x in ent_pos:
            #x => ['Q140258', 0, 12, 'Zagat Survey']
            res = tokenizer.tokenize(" "+x[-1])
            pos = 0
            mark = False
            while res[0] in tokens_a[pos:]:
                idx = tokens_a.index(res[0], pos)
                if check_pre(tokens_a[idx:], res):
                    #entities_a[idx] = x[0]
                    entities[idx] = x[0]
                    mark = True
                    break
                else:
                    pos = idx + 1
            if mark:
                continue
            old_res = res
            res = tokenizer.tokenize(x[-1])
            pos = 0
            while res[0] in tokens_a[pos:]:
                idx = tokens_a.index(res[0], pos)
                if check_pre(tokens_a[idx:], res):
                    #entities_a[idx] = x[0]
                    entities[idx] = x[0]
                    mark = True
                    break
                else:
                    pos = idx + 1
            if not mark:
                print(ex_text_a)
                print("---")
                print(old_res)
                print("---")
                print(res)
                print("---")
                print(tokens_a)
                print("---")
                assert mark
        ######
        ###


        if h[1] == h[2]:
            print("h[1]==h[2]")
            exit()
            continue
        mark = False
        tokens_b = None
        for e in entities_a:
            if e != "UNK":
                mark = True
                break
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
            entities_a = entities_a[:(max_seq_length - 2)]
            entities = entities[:(max_seq_length - 2)]


        #tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        tokens = ["<s>"] + tokens_a + ["</s>"]
        real_ents = ["UNK"] + entities + ["UNK"]
        ents = ["UNK"] + entities_a + ["UNK"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        span_mask = []
        for ent in ents:
            if ent != "UNK":
                span_mask.append(1)
            else:
                span_mask.append(0)



        input_ent = []
        ent_mask = []
        for ent in real_ents:
            if ent != "UNK" and ent in entity2id:
                input_ent.append(entity2id[ent])
                ent_mask.append(1)
            else:
                input_ent.append(-1)
                ent_mask.append(0)
        ent_mask[0] = 1


        if not mark:
            print(example.guid)
            print(example.text_a[0])
            print(example.text_a[0][example.text_a[1][0][1]:example.text_a[1][0][2]])
            print(ents)
            exit(1)
        if sum(span_mask) == 0:
            print("span_mask")
            exit()
            continue

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
            entities_a = entities_a[:(max_seq_length - 2)]

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        padding_ = [-1] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        ent_mask += padding
        input_ent += padding_

        input_ids_ = torch.LongTensor(input_ids)
        if len(input_ids_[input_ids_==50001])!=1 or len(input_ids_[input_ids_==50210])!=1:
            print("Less than 1 'UCHIJ' or 'CLSID' ")
            print(ex_text_a)
            print(tokens_a)
            print(entities)
            exit()


        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(ent_mask) == max_seq_length
        assert len(input_ent) == max_seq_length

        labels = [0]*len(label_map)
        for l in example.label:
            l = label_map[l]
            labels[l] = 1
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("Entity: %s" % example.text_a[1])
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in zip(tokens, ents)]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("label: %s %s" % (example.label, labels))
            logger.info(real_ents)

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              input_ent=input_ent,
                              ent_mask=ent_mask,
                              labels=labels))
        #######
        '''
        print("ex_text_a:")
        print(ex_text_a)
        print("---")
        print("tokens:")
        print(tokens)
        print(len(tokens))
        print("---")
        print("ent_pos:")
        print(ent_pos)
        print("---")
        print("'ĠCLSID':")
        print(tokens.index('ĠCLSID'))
        print("Entity:")
        print(tokens[tokens.index('ĠCLSID'):tokens.index('ĠUCHIJ')+1])
        print("---")
        print("ent_mask:")
        print(ent_mask)
        #print(ent_mask.index(0))
        ent_mask_ = torch.LongTensor(ent_mask)
        print(ent_mask_.nonzero())
        print("---")
        print("input_ent:")
        print(input_ent)
        print("---")
        print("input_mask:")
        print(input_mask)
        print(input_mask.index(0))
        print("---")
        print("input_ids:")
        print(input_ids)
        print(input_mask.index(0))
        print("---")
        print("tokens --> string")
        print("====================")
        print("====================")
        '''
        #######

    return features


def _truncate_seq_pair(tokens_a, tokens_b, ents_a, ents_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            ents_a.pop()
        else:
            tokens_b.pop()
            ents_b.pop()

def accuracy(out, l):
    cnt = 0
    y1 = []
    y2 = []
    for x1, x2 in zip(out, l):
        yy1 = []
        yy2 = []
        top = max(x1)
        for i in range(len(x1)):
            if x1[i] > 0:
                yy1.append(i)
            if x2[i] > 0:
                yy2.append(i)
        y1.append(yy1)
        y2.append(yy2)
        cnt += set(yy1) == set(yy2)
    return cnt, y1, y2

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0






def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--ernie_model", default=None, type=str, required=True,
                        help="Ernie pre-trained model")
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
                        default=3.0,
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
    parser.add_argument('--threshold', type=float, default=.3)
    ##########ADD#######
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
    parser.add_argument('--data_token',
                        type=str,
                        default='None',
                        help="Using token ids")
    ###################

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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    processor = TypingProcessor()

    #tokenizer_label = BertTokenizer_label.from_pretrained(args.ernie_model, do_lower_case=args.do_lower_case)
    #tokenizer = BertTokenizer.from_pretrained(args.ernie_model, do_lower_case=args.do_lower_case)
    tokenizer_label = RobertaTokenizer_label.from_pretrained(args.ernie_model)
    tokenizer = RobertaTokenizer.from_pretrained(args.ernie_model)

    _, label_list, _ = processor.get_train_examples(args.data_dir)
    label_list = sorted(label_list)
    #class_weight = [min(d[x], 100) for x in label_list]
    #logger.info(class_weight)
    S = []
    for l in label_list:
        s = []
        for ll in label_list:
            if ll in l:
                s.append(1.)
            else:
                s.append(0.)
        S.append(s)

    # Prepare model
    filenames = os.listdir(args.output_dir)
    filenames = [x for x in filenames if "pytorch_model.bin_" in x]

    file_mark = []
    for x in filenames:
        file_mark.append([x, True])
        file_mark.append([x, False])

    for x, mark in file_mark:
        print(x, mark)
        output_model_file = os.path.join(args.output_dir, x)
        model_state_dict = torch.load(output_model_file)
        model, _ = BertForEntityTyping.from_pretrained(args.ernie_model, state_dict=model_state_dict, num_labels=len(label_list), args=args)
        model.to(device)
        if args.fp16:
            model.half()

        if mark:
            eval_examples = processor.get_dev_examples(args.data_dir)
            #eval_examples, _, _ = processor.get_train_examples(args.data_dir)
        else:
            eval_examples = processor.get_test_examples(args.data_dir)
            #eval_examples, _, _ = processor.get_train_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer_label, tokenizer, args.threshold)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        # zeros = [0 for _ in range(args.max_seq_length)]
        # zeros_ent = [0 for _ in range(100)]
        # zeros_ent = [zeros_ent for _ in range(args.max_seq_length)]
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_input_ent = torch.tensor([f.input_ent for f in eval_features], dtype=torch.long)
        all_ent_mask = torch.tensor([f.ent_mask for f in eval_features], dtype=torch.long)
        all_labels = torch.tensor([f.labels for f in eval_features], dtype=torch.float)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_input_ent, all_ent_mask, all_labels)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        pred = []
        true = []
        #for input_ids, input_mask, segment_ids, input_ent, ent_mask, labels in eval_dataloader:
        ###
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) if i != 3 else t for i, t in enumerate(batch))
            input_ids, input_mask, segment_ids, input_ent, ent_mask, labels = batch
        ###
            input_ent += 1
            k_1, v_1, k_2, v_2 = load_k_v_queryR_small(input_ent)

            with torch.no_grad():
                #tmp_eval_loss, _ = model(input_ids, segment_ids, input_mask, input_ent, ent_mask, labels, k_1, v_1, k_2, v_2)
                #logits = model(input_ids, segment_ids, input_mask, input_ent, ent_mask, None, k_1, v_1, k_2, v_2)
                if args.fp16:
                    tmp_eval_loss, _ = model(input_ids, segment_ids, input_mask, input_ent, ent_mask, labels.half(), k_1.half(), v_1.half(), k_2.half(), v_2.half())
                    logits = model(input_ids, segment_ids, input_mask, input_ent, ent_mask, None, k_1.half(), v_1.half(), k_2.half(), v_2.half())
                else:
                    tmp_eval_loss, _ = model(input_ids, segment_ids, input_mask, input_ent, ent_mask, labels, k_1, v_1, k_2, v_2)
                    logits = model(input_ids, segment_ids, input_mask, input_ent, ent_mask, None, k_1, v_1, k_2, v_2)


            logits = logits.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()

            tmp_eval_accuracy, tmp_pred, tmp_true = accuracy(logits, labels)

            pred.extend(tmp_pred)
            true.extend(tmp_true)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples


        def f1(p, r):
            if r == 0.:
                return 0.
            return 2 * p * r / float( p + r )
        def loose_macro(true, pred):
            num_entities = len(true)
            p = 0.
            r = 0.
            for true_labels, predicted_labels in zip(true, pred):
                if len(predicted_labels) > 0:
                    p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
                if len(true_labels):
                    r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
            precision = p / num_entities
            recall = r / num_entities
            return precision, recall, f1( precision, recall)
        def loose_micro(true, pred):
            num_predicted_labels = 0.
            num_true_labels = 0.
            num_correct_labels = 0.
            for true_labels, predicted_labels in zip(true, pred):
                num_predicted_labels += len(predicted_labels)
                num_true_labels += len(true_labels)
                num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
            if num_predicted_labels > 0:
                precision = num_correct_labels / num_predicted_labels
            else:
                precision = 0.
            recall = num_correct_labels / num_true_labels
            return precision, recall, f1( precision, recall)


        result = {'eval_loss': eval_loss,
                'eval_accuracy': eval_accuracy,
                'macro': loose_macro(true, pred),
                'micro': loose_micro(true, pred)
                }

        if mark:
            output_eval_file = os.path.join(args.output_dir, "eval_results_{}.txt".format(x.split("_")[-1]))
        else:
            output_eval_file = os.path.join(args.output_dir, "test_results_{}.txt".format(x.split("_")[-1]))
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        #print("==================================")
        #print("==================================")
        #print("==================================")



    exit(0)

if __name__ == "__main__":
    main()
