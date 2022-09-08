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

from knowledge_bert.tokenization import BertTokenizer
from knowledge_bert.modeling_new_n_CLS_comb200 import BertForSequenceClassification
from knowledge_bert.optimization import BertAdam
from knowledge_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

#########
from torch.nn import init
from torch.autograd import Variable
from collections import defaultdict
#from GAT.word_attention_graph import Attention_graph
#########
import time
import pickle
import random






logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

'''
class Model(nn.Module):
    def __init__(self, model_att, model_bert):
        super(self,Model).__init__()
        self.Att = model_att
        self.Bert = model_bert
        self.embed = embed
        self.adj_list = adj_list

    def forward(input_ids, input_mask, segment_ids, input_ent, ent_mask, label_ids, querys, device, fp16)

        #input_ent --> [....256 dim....] -> id and 0
        input_ent = self.embed(input_ent+1).to(device) # -1 -> 0
        #input_ent --->( x,x,x(dim in every []) )

        input_ent = self.Att(ks_vs,querys)

        if fp16:
            loss = Bert(input_ids, segment_ids, input_mask, input_ent.half(), ent_mask, label_ids)
        else:
            loss = Bert(input_ids, segment_ids, input_mask, input_ent, ent_mask, label_ids)

        return loss
'''


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

    def __init__(self, input_ids, input_mask, segment_ids, input_ent, ent_mask, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
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
        with open(input_file, "r", encoding='utf-8') as f:
            return json.loads(f.read())

class FewrelProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")
        labels = set([x.label for x in examples])
        return examples, list(labels)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_labels(self):
        """Useless"""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            for x in line['ents']:
                if x[1] == 1:
                    x[1] = 0
            text_a = (line['text'], line['ents'])
            label = line['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples



def load_ent_emb_static():

    with open('../../data/load_data_n/e1_e2_list_2D_Tensor.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data_test/e1_e2_list_2D_Tensor.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data/e1_e2.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data_test/e1_e2.pkl', 'rb') as f:
        ent_neighbor = pickle.load(f)

    with open('../../data/load_data_n/e1_r_list_2D_Tensor.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data_test/e1_r_list_2D_Tensor.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data/e1_r.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data_test/e1_r.pkl', 'rb') as f:
        ent_r = pickle.load(f)

    with open('../../data/load_data_n/e1_outORin_list_2D_Tensor.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data_test/e1_outORin_list_2D_Tensor.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data/e1_outORin.pkl', 'rb') as f:
    #with open('code/knowledge_bert/load_data_test/e1_outORin.pkl', 'rb') as f:
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




def load_batch_k_v_queryE(input_ent,max_neighbor=3):
    input_ent = input_ent.cpu()
    input_ent_neighbor_emb = torch.zeros(input_ent.shape[0],input_ent.shape[1],max_neighbor).long()
    input_ent_r_emb = torch.zeros(input_ent.shape[0],input_ent.shape[1],max_neighbor).long()
    input_ent_outORin_emb = torch.zeros(input_ent.shape[0],input_ent.shape[1],max_neighbor)

    ent_pos_s = torch.nonzero(input_ent)
    ents = input_ent[input_ent!=0]

    for i,ent in enumerate(ents): #batch_times
        ##########
        #e_neighbor
        ##########
        neighbor_length = len(ent_neighbor[int(ent)])
        if neighbor_length < max_neighbor:
            input_ent_neighbor_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][:] = torch.LongTensor(ent_neighbor[int(ent)]+[0]*(max_neighbor-neighbor_length))

        else:
            input_ent_neighbor_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][:] = torch.LongTensor(ent_neighbor[int(ent)][:max_neighbor])

        ##########
        #e_r
        ##########
        r_length = len(ent_r[int(ent)])
        if r_length < max_neighbor:
            input_ent_r_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][:] = torch.LongTensor(ent_r[int(ent)]+[0]*(max_neighbor-r_length))

        else:
            input_ent_r_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][:] = torch.LongTensor(ent_r[int(ent)][:max_neighbor])


        ##########
        #e_outORin
        ##########
        outORin_length = len(ent_outORin[int(ent)])
        if outORin_length < max_neighbor:
            input_ent_outORin_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][:] = torch.FloatTensor(ent_outORin[int(ent)]+[0]*(max_neighbor-outORin_length))

        else:
            input_ent_outORin_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][:] = torch.FloatTensor(ent_outORin[int(ent)][:max_neighbor])


    #load_embedding
    #e
    input_ent_neighbor_emb = input_ent_neighbor_emb.reshape(input_ent_neighbor_emb.shape[0]*input_ent_neighbor_emb.shape[1],max_neighbor)
    input_ent_neighbor_emb = torch.index_select(embed_ent,0,input_ent_neighbor_emb.reshape(input_ent_neighbor_emb.shape[0]*input_ent_neighbor_emb.shape[1])) #
    input_ent_neighbor_emb = input_ent_neighbor_emb.reshape(input_ent.shape[0],input_ent.shape[1],max_neighbor,100)

    #r
    input_ent_r_emb = input_ent_r_emb.reshape(input_ent_r_emb.shape[0]*input_ent_r_emb.shape[1],max_neighbor)
    input_ent_r_emb = torch.index_select(embed_ent,0,input_ent_r_emb.reshape(input_ent_r_emb.shape[0]*input_ent_r_emb.shape[1])) #
    input_ent_r_emb = input_ent_r_emb.reshape(input_ent.shape[0],input_ent.shape[1],max_neighbor,100)

    #outORin
    input_ent_outORin_emb = input_ent_outORin_emb.unsqueeze(3)

    #Output e_
    k = input_ent_neighbor_emb.cuda()+input_ent_outORin_emb.cuda()*input_ent_r_emb.cuda()
    v = k
    return k,v



def load_batch_k_v_queryR(input_ent,max_neighbor=4): #cannot ramdom --> because of static position
    input_ent = input_ent.cpu()
    input_ent_neighbor_emb = torch.zeros(input_ent.shape[0],input_ent.shape[1],max_neighbor).long()
    input_ent_r_emb = torch.zeros(input_ent.shape[0],input_ent.shape[1],max_neighbor).long()
    input_ent_outORin_emb = torch.zeros(input_ent.shape[0],input_ent.shape[1],max_neighbor)

    ent_pos_s = torch.nonzero(input_ent)
    ents = input_ent[input_ent!=0]

    for i,ent in enumerate(ents): #batch_times
        ##########
        #e_neighbor
        ##########
        neighbor_length = len(ent_neighbor[int(ent)])
        if neighbor_length < max_neighbor:
            input_ent_neighbor_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][:] = torch.LongTensor(ent_neighbor[int(ent)]+[0]*(max_neighbor-neighbor_length))

        else:
            input_ent_neighbor_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][:] = torch.LongTensor(ent_neighbor[int(ent)][:max_neighbor])

        ##########
        #e_r
        ##########
        r_length = len(ent_r[int(ent)])
        if r_length < max_neighbor:
            input_ent_r_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][:] = torch.LongTensor(ent_r[int(ent)]+[0]*(max_neighbor-r_length))

        else:
            input_ent_r_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][:] = torch.LongTensor(ent_r[int(ent)][:max_neighbor])


        ##########
        #e_outORin
        ##########
        outORin_length = len(ent_outORin[int(ent)])
        if outORin_length < max_neighbor:
            input_ent_outORin_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][:] = torch.FloatTensor(ent_outORin[int(ent)]+[0]*(max_neighbor-outORin_length))

        else:
            input_ent_outORin_emb[int(ent_pos_s[i][0])][int(ent_pos_s[i][1])][:] = torch.FloatTensor(ent_outORin[int(ent)][:max_neighbor])


    #load_embedding
    #e
    input_ent_neighbor_emb = input_ent_neighbor_emb.reshape(input_ent_neighbor_emb.shape[0]*input_ent_neighbor_emb.shape[1],max_neighbor)
    #print(input_ent_neighbor_emb.shape)
    input_ent_neighbor_emb = torch.index_select(embed_ent,0,input_ent_neighbor_emb.reshape(input_ent_neighbor_emb.shape[0]*input_ent_neighbor_emb.shape[1])) #
    #print(input_ent_neighbor_emb.shape)
    input_ent_neighbor_emb = input_ent_neighbor_emb.reshape(input_ent.shape[0],input_ent.shape[1],max_neighbor,100)
    #print(input_ent_neighbor_emb.shape)
    #print("===============")

    #r
    input_ent_r_emb = input_ent_r_emb.reshape(input_ent_r_emb.shape[0]*input_ent_r_emb.shape[1],max_neighbor)
    #print(input_ent_r_emb.shape)
    input_ent_r_emb = torch.index_select(embed_ent,0,input_ent_r_emb.reshape(input_ent_r_emb.shape[0]*input_ent_r_emb.shape[1])) #
    #print(input_ent_r_emb.shape)
    input_ent_r_emb = input_ent_r_emb.reshape(input_ent.shape[0],input_ent.shape[1],max_neighbor,100)
    #print(input_ent_r_emb.shape)
    #print("===============")

    #outORin
    #print(input_ent_outORin_emb.shape)
    input_ent_outORin_emb = input_ent_outORin_emb.unsqueeze(3)
    #print(input_ent_outORin_emb.shape)
    #exit()

    #print(input_ent_neighbor_emb)
    #print(input_ent_neighbor_emb.shape)

    #Output e_
    k = input_ent_outORin_emb.cuda()*input_ent_r_emb.cuda()
    v = input_ent_neighbor_emb.cuda()+k
    return k,v
    #k = input_ent_outORin_emb*input_ent_r_emb
    #v = input_ent_neighbor_emb+k
    #return k.cuda(),v.cuda()




def load_k_v_queryR(input_ent):

    #1 line
    #create input_ent_neighbor_emb:
    input_ent = input_ent.cpu()
    input_ent_neighbor_emb = torch.index_select(ent_neighbor,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1])).long()
    #print(input_ent_neighbor_emb.shape)
    input_ent_neighbor_emb = torch.index_select(embed_ent,0,input_ent_neighbor_emb.reshape(input_ent_neighbor_emb.shape[0]*input_ent_neighbor_emb.shape[1])) #
    #print(input_ent_neighbor_emb.shape)
    input_ent_neighbor_emb = input_ent_neighbor_emb.reshape(input_ent.shape[0],input_ent.shape[1],ent_neighbor.shape[1],100)
    #print(input_ent_neighbor_emb.shape)

    #create input_ent_r:
    input_ent_r_emb = torch.index_select(ent_r,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1])).long()
    input_ent_r_emb = torch.index_select(embed_ent,0,input_ent_r_emb.reshape(input_ent_r_emb.shape[0]*input_ent_r_emb.shape[1])) #
    input_ent_r_emb = input_ent_r_emb.reshape(input_ent.shape[0],input_ent.shape[1],ent_r.shape[1],100)

    #create outORin:
    input_ent_outORin_emb = torch.index_select(ent_outORin,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1]))
    #print(input_ent_outORin_emb.shape)
    input_ent_outORin_emb = input_ent_outORin_emb.reshape(input_ent.shape[0],input_ent.shape[1],input_ent_outORin_emb.shape[1])
    #print(input_ent_outORin_emb.shape)
    input_ent_outORin_emb = input_ent_outORin_emb.unsqueeze(3)
    #print(input_ent_outORin_emb.shape)

    #Output e_
    k = input_ent_outORin_emb.cuda()*input_ent_r_emb.cuda()
    v = input_ent_neighbor_emb.cuda()+k
    return k,v



def load_k_v_queryE(input_ent):
    #sentence_word = 256
    #entity_neighbor = 50
    #entity_neighbor = 4

    #1 line
    #create input_ent_neighbor_emb:
    input_ent = input_ent.cpu()
    input_ent_neighbor_emb = torch.index_select(ent_neighbor,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1])).long()
    input_ent_neighbor_emb = torch.index_select(embed_ent,0,input_ent_neighbor_emb.reshape(input_ent_neighbor_emb.shape[0]*input_ent_neighbor_emb.shape[1])) #
    input_ent_neighbor_emb = input_ent_neighbor_emb.reshape(input_ent.shape[0],input_ent.shape[1],ent_neighbor.shape[1],100)

    #create input_ent_r:
    input_ent_r_emb = torch.index_select(ent_r,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1])).long()
    input_ent_r_emb = torch.index_select(embed_ent,0,input_ent_r_emb.reshape(input_ent_r_emb.shape[0]*input_ent_r_emb.shape[1])) #
    input_ent_r_emb = input_ent_r_emb.reshape(input_ent.shape[0],input_ent.shape[1],ent_r.shape[1],100)

    #create outORin:
    input_ent_outORin_emb = torch.index_select(ent_outORin,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1]))
    input_ent_outORin_emb = input_ent_outORin_emb.reshape(input_ent.shape[0],input_ent.shape[1],input_ent_outORin_emb.shape[1])
    input_ent_outORin_emb = input_ent_outORin_emb.unsqueeze(3)

    #Output e_
    k = input_ent_neighbor_emb.cuda()+input_ent_outORin_emb.cuda()*input_ent_r_emb.cuda()
    v = k
    return k,v



def load_k_v_queryR_small(input_ent):
        input_ent = input_ent.cpu()


        ent_pos_s = torch.nonzero(input_ent)

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
        #print("============")
        #exit()


        #Neighbor
        input_ent_neighbor = torch.index_select(ent_neighbor,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1])).long()
        #print(input_ent_neighbor)
        #print(input_ent_neighbor.shape)
        #print("===")

        #create input_ent_neighbor_1
        input_ent_neighbor_emb_1 = torch.index_select(embed_ent,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1])) #
        input_ent_neighbor_emb_1 = input_ent_neighbor_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],ent_neighbor.shape[1],embed_ent.shape[-1])

        #create input_ent_r_1:
        input_ent_r_emb_1 = torch.index_select(ent_r,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1])).long()
        input_ent_r_emb_1 = torch.index_select(embed_r,0,input_ent_r_emb_1.reshape(input_ent_r_emb_1.shape[0]*input_ent_r_emb_1.shape[1])) #
        input_ent_r_emb_1 = input_ent_r_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],ent_r.shape[1],embed_r.shape[-1])

        #create outORin_1:
        input_ent_outORin_emb_1 = torch.index_select(ent_outORin,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1]))
        input_ent_outORin_emb_1 = input_ent_outORin_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],input_ent_outORin_emb_1.shape[1])
        input_ent_outORin_emb_1 = input_ent_outORin_emb_1.unsqueeze(3)


        #create input_ent_neighbor_2
        input_ent_neighbor_2 = torch.index_select(ent_neighbor,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1])).long()
        #print("----")
        #print(input_ent_neighbor_2)
        #print(input_ent_neighbor_2.shape)
        #print("----")
        #print("===")
        input_ent_neighbor_emb_2 = torch.index_select(embed_ent,0,input_ent_neighbor_2.reshape(input_ent_neighbor_2.shape[0]*input_ent_neighbor_2.shape[1])) #
        input_ent_neighbor_emb_2 = input_ent_neighbor_emb_2.reshape(input_ent.shape[0],input_ent.shape[1],ent_neighbor.shape[1],ent_neighbor.shape[1],embed_ent.shape[-1])
        #print(input_ent_neighbor_emb_2)
        #print(input_ent_neighbor_emb_2.shape)
        #print("===")


        #create input_ent_r_2:
        input_ent_r_2 = torch.index_select(ent_r,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1])).long()
        #print(input_ent_r_2)
        #print(input_ent_r_2.shape)
        #print("====")
        input_ent_r_emb_2 = torch.index_select(embed_r,0,input_ent_r_2.reshape(input_ent_r_2.shape[0]*input_ent_r_2.shape[1])) #
        input_ent_r_emb_2 = input_ent_r_emb_2.reshape(input_ent.shape[0],input_ent.shape[1],ent_r.shape[1],ent_neighbor.shape[1],embed_r.shape[-1])
        #print(input_ent_r_emb_2)
        #print(input_ent_r_emb_2.shape)
        #print("====")

        #create outORin_2: #?
        input_ent_outORin_emb_2 = torch.index_select(ent_outORin,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1]))
        #print(input_ent_outORin_emb_2)
        #print(input_ent_outORin_emb_2.shape)
        #print("====")
        input_ent_outORin_emb_2 = input_ent_outORin_emb_2.reshape(input_ent_r_emb_2.shape[0],input_ent_r_emb_2.shape[1],input_ent_r_emb_2.shape[2],input_ent_r_emb_2.shape[3])
        input_ent_outORin_emb_2 = input_ent_outORin_emb_2.unsqueeze(4)
        #print(input_ent_outORin_emb_2)
        #print(input_ent_outORin_emb_2.shape)
        #print("====")

        ###
        k_1 = input_ent_outORin_emb_1.cuda()*input_ent_r_emb_1.cuda()
        v_1 = input_ent_neighbor_emb_1.cuda()+k_1
        k_2 = input_ent_outORin_emb_2.cuda()*input_ent_r_emb_2.cuda()
        #print(k_2)
        #print(k_2.shape)
        #print("===")
        #print(input_ent_outORin_emb_2.shape)
        #print(input_ent_r_emb_2.shape)
        v_2 = input_ent_neighbor_emb_2.cuda()+k_2
        #print(v_2.shape)
        #exit()
        return k_1,v_1,k_2,v_2


print("Load Emb ...")
embed_ent, embed_r = load_knowledge()
ent_neighbor, ent_r, ent_outORin = load_ent_emb_static()
#ent_neighbor, ent_r, ent_outORin = load_ent_emb_dynamic()
print("Finsh loading Emb")




###
#def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, threshold, adj_list, embed):
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, threshold):
    """Loads a data file into a list of `InputBatch`s."""

    label_list = sorted(label_list)
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
        h, t = example.text_a[1]
        h_name = ex_text_a[h[1]:h[2]]
        t_name = ex_text_a[t[1]:t[2]]
        # Add [HD] and [TL], which are "#" and "$" respectively.
        if h[1] < t[1]:
            ex_text_a = ex_text_a[:h[1]] + "∞ "+h_name+" π" + ex_text_a[h[2]:t[1]] + "º "+t_name+" ∂" + ex_text_a[t[2]:]
        else:
            ex_text_a = ex_text_a[:t[1]] + "º "+t_name+" ∂" + ex_text_a[t[2]:h[1]] + "∞ "+h_name+" π" + ex_text_a[h[2]:]

        if h[1] < t[1]:
            h[1] += 2
            h[2] += 2
            t[1] += 6
            t[2] += 6
        else:
            h[1] += 6
            h[2] += 6
            t[1] += 2
            t[2] += 2

        tokens_a, entities_a = tokenizer.tokenize(ex_text_a, [h, t])
        #print(tokens_a)
        #print("=======")
        #print(tokenizer.convert_tokens_to_ids(tokens_a))
        #exit()

        if len([x for x in entities_a if x!="UNK"]) != 2:
            #print(entities_a, len([x for x in entities_a if x[0]!="UNK"]))
            exit(1)

        tokens_b = None
        if example.text_b:
            tokens_b, entities_b = tokenizer.tokenize(example.text_b[0], [x for x in example.text_b[1] if x[-1]>threshold])
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, entities_a, entities_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
                entities_a = entities_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        ents = ["UNK"] + entities_a + ["UNK"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            ents += entities_b + ["UNK"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ent = []
        ent_mask = []
        ###
        #nodes = []
        for ent in ents:
            if ent != "UNK" and ent in entity2id:
                input_ent.append(entity2id[ent])
                ###
                #nodes.append(adj_list[entity2id[ent]])
                ent_mask.append(1)
            else:
                input_ent.append(-1)
                ent_mask.append(0)
        ent_mask[0] = 1
        ###
        #print(nodes)
        #exit()

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        padding_ = [-1] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        input_ent += padding_
        ent_mask += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(input_ent) == max_seq_length
        assert len(ent_mask) == max_seq_length

        label_id = label_map[example.label]

        if input_ids.count(1601)!= 1 or input_ids.count(1089)!=1:
            print(tokens_a)
            print("---")
            print(input_ids)
            print("---")
            print("∞:",input_ids.count(1601),";","º:",input_ids.count(1089))
            print("1601:",input_ids.count(1601),";","º:",input_ids.count(1089))
            print("=======")


        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              input_ent=input_ent,
                              ent_mask=ent_mask,
                              label_id=label_id))
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

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

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

    processors = FewrelProcessor

    num_labels_task = 80

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

    if not args.do_train:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)


    processor = processors()
    num_labels = num_labels_task
    label_list = None

    tokenizer = BertTokenizer.from_pretrained(args.ernie_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    train_examples, label_list = processor.get_train_examples(args.data_dir)
    num_train_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    #Load_KG_emb
    #embed, adj_list = load_knowledge()

    # Prepare att model
    #model_att = Attention_graph(k_v_features=embed, k_v_dim=100, q_dim=256, adj_lists=adj_list, batch_size=args.train_batch_size, self_att=True, cuda=True, device=device)


    # Prepare model
    model, _ = BertForSequenceClassification.from_pretrained(args.ernie_model,
              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
              num_labels = num_labels, args=args)


    #Fix here
    #model = Model(model_att, model_bert, embed, adj_list)

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
    #print(param_optimizer)
    #print(model.state_dict())
    #exit()
    no_grad = ['bert.encoder.layer.11.output.dense_ent', 'bert.encoder.layer.11.output.LayerNorm_ent']
    #no_grad = ['bert.encoder.layer.11.output.dense_ent', 'bert.encoder.layer.11.output.LayerNorm_ent','bert.word_graph_attention.Q_linear_2','bert.word_graph_attention.K_V_linear_2','bert.word_attention_graph.Q_linear_1','bert.word_attention_graph.K_V_linear_1']
    param_optimizer = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_grad)]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
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

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
    global_step = 0
    if args.do_train:
        ###
        #train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer, args.threshold, adj_list, embed)
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer, args.threshold)


        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_ent = torch.tensor([f.input_ent for f in train_features], dtype=torch.long)
        all_ent_masks = torch.tensor([f.ent_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_ent, all_ent_masks, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        output_loss_file = os.path.join(args.output_dir, "loss")
        loss_fout = open(output_loss_file, 'w')
        model.train()


        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) if i != 3 else t for i, t in enumerate(batch))
                input_ids, input_mask, segment_ids, input_ent, ent_mask, label_ids = batch

                input_ent = input_ent+1

                k_1, v_1, k_2, v_2 = load_k_v_queryR_small(input_ent)

                loss=0
                if args.fp16:
                    loss = model(input_ids, segment_ids, input_mask, input_ent.float(), ent_mask, label_ids, k_1.half(), v_1.half(), k_2.half(), v_2.half())


                else:
                    loss = model(input_ids, segment_ids, input_mask, input_ent.float(), ent_mask, label_ids, k_1, v_1, k_2, v_2)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                loss_fout.write("{}\n".format(loss.item()))
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
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(args.output_dir, "pytorch_model.bin_{}".format(global_step))
            torch.save(model_to_save.state_dict(), output_model_file)

        # Save a trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)

if __name__ == "__main__":
    main()
