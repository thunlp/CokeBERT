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
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn import functional as F
from .file_utils import cached_path

#########
from torch.nn import init
from torch.autograd import Variable
from collections import defaultdict
import numpy as np
#########
import pickle

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'ernie_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_types=None):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_types = layer_types
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    class BertLayerNorm(nn.Module):
        #def __init__(self, hidden_size, eps=1e-12):
        def __init__(self, hidden_size, eps=config.layer_norm_eps):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        #self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        #print(mixed_query_layer)
        #print(mixed_query_layer.shape)
        #print(mixed_key_layer)
        #print(mixed_key_layer.shape)
        #print(mixed_value_layer)
        #print(mixed_value_layer.shape)
        #exit()

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        #print(attention_mask)
        #print(attention_mask.shape)
        #print("-----")
        #print(attention_scores)
        #print(attention_scores.shape)
        #print("-----")
        attention_scores = attention_scores + attention_mask
        #print("-----")
        #print(attention_scores)
        #print(attention_scores.shape)
        #exit()

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        #print(attention_probs)
        #print(attention_probs.shape)
        #print(value_layer)
        #print(value_layer.shape)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

        config_ent = copy.deepcopy(config)
        config_ent.hidden_size = 100
        config_ent.num_attention_heads = 4

        self.self_ent = BertSelfAttention(config_ent)
        self.output_ent = BertSelfOutput(config_ent)


    def forward(self, input_tensor, attention_mask, input_tensor_ent, attention_mask_ent):
        self_output = self.self(input_tensor, attention_mask)
        self_output_ent = self.self_ent(input_tensor_ent, attention_mask_ent)
        attention_output = self.output(self_output, input_tensor)
        attention_output_ent = self.output_ent(self_output_ent, input_tensor_ent)
        return attention_output, attention_output_ent

class BertAttention_simple(nn.Module):
    def __init__(self, config):
        super(BertAttention_simple, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        #self.dense_ent = nn.Linear(100, config.intermediate_size)
        self.dense_ent = nn.Linear(100, config.intermediate_size,bias=False)

        #self.dense_1 = nn.Linear(config.hidden_size, 200)
        #self.dense_1_ent = nn.Linear(100, 200)

        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states, hidden_states_ent):
        #print(hidden_states_ent)
        #print(hidden_states_ent.shape)
        hidden_states_ = self.dense(hidden_states)
        #print(hidden_states_ent)
        #print(hidden_states_ent.shape)
        #print("--------")
        hidden_states_ent_ = self.dense_ent(hidden_states_ent)
        #print(hidden_states_ent_)
        #print(hidden_states_ent_.shape)
        #print(hidden_states)
        #print(hidden_states.shape)
        #exit()

        #hidden_states_1 = self.dense_1(hidden_states)
        #hidden_states_ent_1 = self.dense_1_ent(hidden_states_ent)

        hidden_states = self.intermediate_act_fn(hidden_states_+hidden_states_ent_)
        #hidden_states_ent = self.intermediate_act_fn(hidden_states_1+hidden_states_ent_1)

        return hidden_states#, hidden_states_ent

class BertIntermediate_simple(nn.Module):
    def __init__(self, config):
        super(BertIntermediate_simple, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense_ent = nn.Linear(config.intermediate_size, 100)
        #self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        #self.LayerNorm_ent = BertLayerNorm(100, eps=1e-12)
        self.LayerNorm_ent = BertLayerNorm(100, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states_, input_tensor, input_tensor_ent):
        hidden_states = self.dense(hidden_states_)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        hidden_states_ent = self.dense_ent(hidden_states_)

        #print("372")
        #print("hidden_states_ent")
        #print(hidden_states_ent)
        #print(hidden_states_ent.shape)

        hidden_states_ent = self.dropout(hidden_states_ent)
        hidden_states_ent = self.LayerNorm_ent(hidden_states_ent + input_tensor_ent)

        return hidden_states, hidden_states_ent


class BertOutput_simple(nn.Module):
    def __init__(self, config):
        super(BertOutput_simple, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        #self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class BertLayerMix(nn.Module):
    def __init__(self, config):
        super(BertLayerMix, self).__init__()
        self.attention = BertAttention_simple(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, hidden_states_ent, attention_mask_ent, ent_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output_ent = hidden_states_ent * ent_mask
        intermediate_output = self.intermediate(attention_output, attention_output_ent)
        layer_output, layer_output_ent = self.output(intermediate_output, attention_output, attention_output_ent)
        # layer_output_ent = layer_output_ent * ent_mask
        return layer_output, layer_output_ent

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, hidden_states_ent, attention_mask_ent, ent_mask):
        attention_output, attention_output_ent = self.attention(hidden_states, attention_mask, hidden_states_ent, attention_mask_ent)

        #print("434")
        #print("attention_output_ent")
        #print(attention_output_ent)
        #print(attention_output_ent.shape)
        #print(ent_mask)
        #print(ent_mask.shape)
        #exit()

        attention_output_ent = attention_output_ent * ent_mask
        intermediate_output = self.intermediate(attention_output, attention_output_ent)
        layer_output, layer_output_ent = self.output(intermediate_output, attention_output, attention_output_ent)
        # layer_output_ent = layer_output_ent * ent_mask
        return layer_output, layer_output_ent

class BertLayer_simple(nn.Module):
    def __init__(self, config):
        super(BertLayer_simple, self).__init__()
        self.attention = BertAttention_simple(config)
        self.intermediate = BertIntermediate_simple(config)
        self.output = BertOutput_simple(config)

    def forward(self, hidden_states, attention_mask, hidden_states_ent, attention_mask_ent, ent_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, hidden_states_ent



#Fix here
class BertEncoder(nn.Module):
    def __init__(self, config, args=None):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        layer_simple = BertLayer_simple(config)
        layer_mix = BertLayerMix(config)
        layers = []
        for t in config.layer_types:
            if t == "sim":
                layers.append(copy.deepcopy(layer_simple))
            if t == "norm":
                layers.append(copy.deepcopy(layer))
            if t == "mix":
                layers.append(copy.deepcopy(layer_mix))
        for _ in range(config.num_hidden_layers-len(layers)):
            layers.append(copy.deepcopy(layer_simple))
        self.layer = nn.ModuleList(layers)
        ####
        self.layer_types = config.layer_types
        ####
        self.K_V_dim = args.K_V_dim

    def forward(self, hidden_states, all_encoder_layers, attention_mask, hidden_states_ent, attention_mask_ent, ent_mask, i_th_layer=0, output_all_encoded_layers=True, mode=None):

        #all_encoder_layers = all_encoder_layers
        #ent_mask = ent_mask.to(dtype=next(self.parameters()).dtype).unsqueeze(-1)

        if mode=="encoder1":
            sentence_word = hidden_states_ent.shape[1]

            if hidden_states.type() == "torch.cuda.HalfTensor":
                hidden_states_ent = torch.zeros(hidden_states_ent.shape[0], sentence_word, self.K_V_dim).half().cuda()
            else:
                hidden_states_ent = torch.zeros(hidden_states_ent.shape[0], sentence_word, self.K_V_dim).cuda()

            #print("====")
            for i_th_layer,layer_module in enumerate(self.layer):
                #print(i_th_layer,self.layer_types[i_th_layer])

                ###Add####
                if self.layer_types[i_th_layer] == "mix":
                    return hidden_states, all_encoder_layers, i_th_layer

                hidden_states, hidden_states_ent = layer_module(hidden_states, attention_mask, hidden_states_ent, attention_mask_ent, ent_mask)

                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)

        elif mode=="encoder2":

            #print("====")
            for i_th_layer,layer_module in enumerate(self.layer[5:]):
                #print(i_th_layer,self.layer_types[5+i_th_layer])
                hidden_states, hidden_states_ent = layer_module(hidden_states, attention_mask, hidden_states_ent, attention_mask_ent, ent_mask)

                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            if not output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

            #exit()
            return hidden_states, all_encoder_layers, None

        else:
            print("WRONG!!")
            exit()



class Word_Graph_Attention(nn.Module):
    def __init__(self, k_v_dim, q_dim, graphsage=False, self_att=True):
        super(Word_Graph_Attention,self).__init__()
        ##emb, entity
        #embed, self.entity_list = load_knowledge()
        #self.embed = embed.half().cuda()
        #self.embed = embed
        #self.embed = embed.cuda()
        self.K_V_dim = k_v_dim
        self.Q_dim = q_dim
        ###
        #self.K_weight = nn.Parameter(torch.FloatTensor(self.K_V_dim, self.K_V_dim))
        self.K_V_linear = torch.nn.Linear(self.K_V_dim,self.K_V_dim,bias=False)
        init.xavier_uniform_(self.K_V_linear.weight)
        #init.xavier_uniform_(self.K_weight)
        #nn.init.zeros_(self.K_weight)

        ###
        self.V_linear = torch.nn.Linear(self.K_V_dim,self.K_V_dim,bias=False)
        init.xavier_uniform_(self.V_linear.weight)


        ###
        #self.Q_weight = nn.Parameter(torch.FloatTensor(self.K_V_dim, self.Q_dim))
        self.Q_linear = torch.nn.Linear(self.Q_dim,self.K_V_dim,bias=True)
        init.xavier_uniform_(self.Q_linear.weight)
        #init.xavier_uniform_(self.Q_weight)
        #nn.init.zeros_(self.Q_weight)

        self.graphsage = graphsage
        self.self_att = self_att
        #if couda
        self.softmax_0 = nn.Softmax(dim=0)
        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)
        self.LeakyReLU = nn.LeakyReLU()
        self.Tanh = nn.Tanh()
        ###
        #self.att_weight = nn.Parameter(torch.FloatTensor(1, self.K_V_dim*2))
        #init.xavier_uniform_(self.att_weight)
        #nn.init.zeros_(self.att_weight)

        if self.graphsage:
            #self.weight = nn.Parameter(torch.FloatTensor(self.K_V_dim, self.K_V_dim*2))
            #init.xavier_uniform_(self.weight)
            #self.weight.data.copy_( torch.cat( (torch.eye(self.K_V_dim),torch.zeros(self.K_V_dim,self.K_V_dim)), dim=1) )

            self.graphsage_linear = torch.nn.Linear(self.K_V_dim*2, self.K_V_dim, bias=False)
            init.xavier_uniform_(self.graphsage_linear.weight)

        #self.ent_neighbor, self.ent_r, self.ent_outORin = load_ent_emb()
        #self.embed_ent, self.embed_r= load_knowledge()


    #def self_attention(self, e_neighs, query):
    def self_attention(self, Q, K, V):


        ####Matrix####

        ###################
        ####L1 distance####
        '''
        #Q
        Q = self.Q_linear(Q)
        Q = self.Tanh(Q)
        #[,100]
        Q = Q.unsqueeze(1)
        Q = Q.unsqueeze(2)
        Q = Q.repeat(1,K.shape[1],K.shape[2],1)

        #K
        K = self.K_V_linear(K)

        #V
        V = self.V_linear(V) #1. Original V  2.self.K_V_linear(V) 3.self.V_linear(V)

        #attention
        #print(Q.shape)
        #[2,2,100,100] ???
        attention = torch.norm(Q-K, p=1, dim=3, keepdim=True)
        mask = K.sum(dim=3)
        mask[mask!=0] = 1
        mask = mask.unsqueeze(3)
        #attention = mask*attention
        attention = attention + float(1e-6)
        attention = mask.div(attention) #the more l1 loss, the more difference
        attention = attention.masked_fill(attention==0, float('-10000'))
        attention = self.softmax_2(self.LeakyReLU(attention))
        #attention = self.softmax_2(self.LeakyReLU(attention))
        #print(attention)
        #print(attention.shape)
        attention = attention.squeeze(3).unsqueeze(2)

        sentence_entity_reps = attention.matmul(V).squeeze(2)
        '''
        ###################
        ###################

        ####Best###########
        ###################
        #[--]
        #Q

        #[--]
        #print("====")
        Q = self.Q_linear(Q)
        #exit()
        Q = self.Tanh(Q)
        #[--]

        #normolize
        #Q_l2 = F.normalize(Q, p=2, dim=1)
        #Q_l2 = Q_l2.unsqueeze(1)
        #Q_l2 = Q_l2.unsqueeze(2)

        #no normalize
        #Q_l2 = Q.unsqueeze(1)
        #Q_l2 = Q_l2.unsqueeze(2)
        Q_l2 = Q.unsqueeze(1).unsqueeze(2)


        #Q query V:
        K = self.K_V_linear(K)
        #K = self.K_V_linear(V)
        #K = self.V_linear(K)
        #K = seolf.V_linear(V)

        #normolize
        #l2 = torch.norm(K, p=2, dim=3, keepdim=True).detach()
        #l2[l2==0]=float(1e-6)
        #K_l2 = K.div(l2)

        #no normalize
        K_l2 = K



        #normolize
        #attention = (Q_l2*K_l2).sum(3)

        #no normalize
        #print("====")
        #print(Q_l2.shape)
        #print("====")
        #print(K_l2.shape)
        try:
            attention = ((Q_l2*K_l2).sum(3)).div(math.sqrt(self.K_V_dim))
        except:
            print(Q.shape)
            print("===")
            print(Q_l2.shape)
            print("===")
            print(K_l2.shape)
            print("===")
            exit()


        attention = attention.masked_fill(attention==0, float('-10000'))
        attention = self.softmax_2(self.LeakyReLU(attention))
        attention = attention.masked_fill(attention==float(1/attention.shape[-1]), float(0)) # don't need to

        attention = attention.unsqueeze(2)

        #use nn
        V = self.V_linear(V) #1. Original V  2.self.K_V_linear(V) 3.self.V_linear(V)

        #use org V
        #V = V
        try:
            sentence_entity_reps = attention.matmul(V).squeeze(2)
        except:
            print("LINE:717")
            print(attention.shape)
            print(V.shape)
            exit()
        ###################
        ###################


        #########
        ####Avg####
        '''
        mask = V.sum(dim=3)
        mask[mask!=0]=1
        mask[:,:,0]=0
        #print(mask)
        #print(mask.shape)
        try:
            mask[0][1][:]=0
        except:
            mask = torch.zeros(8,2,100).cuda().half()
            print(mask)
            print(mask.shape)
        attention = mask.masked_fill(mask==0, float('-10000'))
        attention = self.softmax_2(attention)
        #attention = attention.masked_fill(attention==float(1/attention.shape[-1]), float(0)) #[1,1,1,1] fail --> don't need to
        attention = attention.unsqueeze(2)
        sentence_entity_reps = attention.matmul(V).squeeze(2)
        #print(sentence_entity_reps)
        #print(sentence_entity_reps.shape)
        #sentence_entity_reps = V[:,:,0,:]
        '''
        #########
        #########

        return sentence_entity_reps




    #def forward(self, sentence_input_ent, q, candidate, types):
    def forward(self, input_ent, q, k, v, mode):
        #print("!!!")
        #print(mode)
        #print("!!!")

        '''
        if mode=="candidate_neg":
            v = v[:,:,0,:]
            v = self.V_linear(v).squeeze(2)
            return v

        else:
            q = q[:,0,:] #all input: 0, !=0
            neigh_feats = self.self_attention(q, k, v)


        ##Agg #v should pass through self.V_linear (neigh_feats,v shuld be same emb space)
        if self.graphsage:
            v = v[:,:,0,:]
            #v = self.V_linear(v)
            combined = torch.cat([v, neigh_feats], dim=2)

            #combined = self.LeakyReLU(self.weight.mm(combined.T).T)
            combined = self.graphsage_linear(combined)
        else:
            combined = neigh_feats

        #return combined

        if mode == "entity":
            #return hidden_states_ent
            hidden_states_ent = torch.zeros(input_ent.shape[0], input_ent.shape[1], self.K_V_dim).half().cuda()

            ent_pos_s = torch.nonzero(input_ent) # id start from 0

            #####
            #ent_pos_s_x = ent_pos_s[:,0]
            #ent_pos_s_y = ent_pos_s[:,1]
            #combined_cord = torch.nonzero(combined)
            #combined_cord = torch.unique(combined_cord[:,[0,1]],dim=0)
            #combined_cord_x = combined_cord[:,0]
            #combined_cord_y = combined_cord[:,1]
            #hidden_states_ent[ent_pos_s_x,ent_pos_s_y] = combined[combined_cord_x,combined_cord_y]
            #####

            for batch in range(input_ent.shape[0]):
                for i,index in enumerate(ent_pos_s[ent_pos_s[:,0]==batch]):
                    hidden_states_ent[batch][int(index[1])] = combined[batch][i]

        elif mode == "candidate_pos":
            hidden_states_ent = combined

        else:
            print("Graph attention Wrong!!")
            exit()

        return hidden_states_ent
        '''


        if mode == "entity":
            #q = q[:,0,:] #all input: 0, !=0
            neigh_feats = self.self_attention(q, k, v)

            if self.graphsage:
                v = v[:,:,0,:]
                v = self.V_linear(v).squeeze(2)
                combined = torch.cat([v, neigh_feats], dim=2)
                combined = self.graphsage_linear(combined)
            else:
                combined = neigh_feats

            #return hidden_states_ent
            hidden_states_ent = torch.zeros(input_ent.shape[0], input_ent.shape[1], self.K_V_dim).half().cuda()
            ent_pos_s = torch.nonzero(input_ent) # id start from 0

            for batch in range(input_ent.shape[0]):
                for i,index in enumerate(ent_pos_s[ent_pos_s[:,0]==batch]):
                    hidden_states_ent[batch][int(index[1])] = combined[batch][i]


        elif mode=="candidate_neg":
            if self.graphsage:
                v = v[:,:,0,:]
                neigh_feats = self.V_linear(v).squeeze(2)
                combined = neigh_feats
                #combined = torch.cat([v, neigh_feats], dim=2)
                #combined = self.graphsage_linear(combined)
            else:
                v = v[:,:,0,:]
                combined = self.V_linear(v).squeeze(2)
                #combined = v
            hidden_states_ent = combined

        elif mode == "candidate_pos":
            #q = q[:,0,:] #all input: 0, !=0
            neigh_feats = self.self_attention(q, k, v)
            if self.graphsage:
                v = v[:,:,0,:]
                v = self.V_linear(v).squeeze(2)
                combined = torch.cat([v, neigh_feats], dim=2)
                combined = self.graphsage_linear(combined)
            else:
                combined = neigh_feats
            hidden_states_ent = combined

        else:
            print("Graph attention mode Wrong!!")
            print(mode)
            exit()

        return hidden_states_ent




class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        ###
        '''
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
        '''
        ###
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(768, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        #self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertEntPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertEntPredictionHead, self).__init__()
        config_ent = copy.deepcopy(config)
        config_ent.hidden_size = 100
        self.transform = BertPredictionHeadTransform(config_ent)
        #pass paras
        #self.word_graph_attention = Word_Graph_Attention(100,768,False,True)


    #def forward(self, hidden_states, candidate):
    def forward(self, hidden_states, candidate_emb):
        #print(candidate) #([[....index...]],256) ==>(1,n,256)
        #print(candidate_emb.shape)
        #exit()

        '''
        k_cand = torch.cat(hidden_states_6th_original.shape[0]*[k_cand])
        v_cand = torch.cat(hidden_states_6th_original.shape[0]*[v_cand])
        cand_pos_tensor = cand_pos_tensor.float().half().unsqueeze(2).unsqueeze(3)
        k_cand_pos = k_cand*cand_pos_tensor
        v_cand_pos = v_cand*cand_pos_tensor

        candidate_pos = self.word_graph_attention(candidate, hidden_states_6th_original[:,0,:], k_cand_pos, v_cand_pos, "candidate_pos") #

        cand_pos_tensor = ((cand_pos_tensor-1)*(-1))
        v_cand_neg = v_cand*cand_pos_tensor
        candidate_neg = self.word_graph_attention(candidate, hidden_states_6th_original[:,0,:], None, v_cand, "candidate_neg") #

        candidate = candidate_pos + candidate_neg
        '''

        hidden_states = self.transform(hidden_states)

        #======================
        #fix below!

        #print(hidden_states)
        #print(hidden_states.shape)
        #print("-----")
        #exit()

        #candidate = torch.zeros(1,9,100).half().cuda()
        #print(candidate)
        #print(candidate.shape)
        #print("-----")
        #candidate = torch.squeeze(candidate, 0)
        #print(candidate)
        #print(candidate.shape)
        #print("-----")
        #exit()
        #print("relation score")
        #print(torch.matmul(hidden_states, candidate.t()))
        #print(torch.matmul(hidden_states, candidate.t()).shape)
        #print(torch.matmul(hidden_states, candidate.transpose(1,2)))
        #print(torch.matmul(hidden_states, candidate.transpose(1,2)).shape)
        #exit()

        # hidden_states [batch_size, max_seq, dim]
        # candidate [entity_num_in_the_batch, dim]
        # return [batch_size, max_seq, entity_num_in_the_batch]
        #return torch.matmul(hidden_states, candidate.t())
        #return torch.matmul(hidden_states, candidate.transpose(1,2))
        return torch.matmul(hidden_states, candidate_emb.transpose(1,2))
        #will return torch.Size([2, 256, 9])


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.predictions_ent = BertEntPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
        #embed,_ = load_knowledge() ##
        #self.embed = torch.nn.Embedding.from_pretrained(embed) ##

    def forward(self, sequence_output, pooled_output, candidate_emb):
    #def forward(self, sequence_output, pooled_output, candidate, hidden_states_6th_original):
        prediction_scores = self.predictions(sequence_output)
        #print(sequence_output.shape)
        #print(prediction_scores.shape)

        seq_relationship_score = self.seq_relationship(pooled_output) #predict last layer CLS token relation type
        #candidate = self.embed(candidate) ##
        #prediction_scores_ent = self.predictions_ent(sequence_output, candidate)
        prediction_scores_ent = self.predictions_ent(sequence_output, candidate_emb)
        return prediction_scores, seq_relationship_score, prediction_scores_ent

#Fix here
class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config


    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, *inputs, **kwargs):
    #def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, *inputs, **kwargs, args=None):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-base-multilingual`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            archive_file = pretrained_model_name
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(archive, tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.

        model = cls(config, *inputs, **kwargs)
        #model = cls(config, *inputs, **kwargs, args=None)

        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        #exit()

        #print(model.parameters()) #will list not being initlized paras
        #embed_ent, embed_r = load_knowledge()
        #model.bert.encoder.embed_ent = nn.Embedding.from_pretrained(embed_ent)
        #model.bert.encoder.embed_ent.weight.requires_grad = False
        #model.bert.encoder.embed_r = nn.Embedding.from_pretrained(embed_r)
        #model.bert.encoder.embed_r.weight.requires_grad = False

        return model, missing_keys


class BertModel(PreTrainedBertModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, args=None):
        super(BertModel, self).__init__(config, args=None)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config,args)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)
        self.word_graph_attention = Word_Graph_Attention(args.K_V_dim,args.Q_dim,args.graphsage,args.self_att)

    #def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, output_all_encoded_layers=True):
    ###
    #def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, output_all_encoded_layers=True, subgraph=None, subgraph_id_map_rep=None):
    #def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, output_all_encoded_layers=True, input_ent_emb=None):


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, output_all_encoded_layers=True, k=None, v=None, candidate=None, k_cand=None, v_cand=None, cand_pos_tensor=None, mode=None):
        #print(cand_pos_tensor)
        #print(cand_pos_tensor.shape)
        #print(candidate)
        #print("=======")
        #print(mode)
        #exit()

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_ent_mask = ent_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_ent_mask = extended_ent_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_ent_mask = (1.0 - extended_ent_mask) * -10000.0

        #############################################################

        embedding_output = self.embeddings(input_ids, token_type_ids)

        ##
        all_encoder_layers=list()
        ent_mask = ent_mask.to(dtype=next(self.parameters()).dtype).unsqueeze(-1)
        ##


        hidden_states, all_encoder_layers, i_th_layer = self.encoder(embedding_output, all_encoder_layers, extended_attention_mask, input_ent, extended_ent_mask, ent_mask, i_th_layer=0,output_all_encoded_layers=output_all_encoded_layers, mode="encoder1")


        if mode=="pretrain":
            last_token = hidden_states[input_ids == 2]
        else:
            last_token = hidden_states[input_ids == 2]

        if len(input_ent[input_ent!=0]) == 0:
            hidden_states_ent = torch.zeros(input_ent.shape[0], input_ent.shape[1],100).cuda().half()
        else:
            hidden_states_ent = self.word_graph_attention(input_ent, last_token, k, v, "entity")

        _, encoded_layers, _ = self.encoder(hidden_states, all_encoder_layers, extended_attention_mask,hidden_states_ent, extended_ent_mask, ent_mask, i_th_layer=i_th_layer, output_all_encoded_layers=output_all_encoded_layers, mode="encoder2")


        if mode=="pretrain":
            k_cand = torch.cat(hidden_states.shape[0]*[k_cand])
            v_cand = torch.cat(hidden_states.shape[0]*[v_cand])
            cand_pos_tensor = cand_pos_tensor.float().half().unsqueeze(2).unsqueeze(3)
            k_cand_pos = k_cand*cand_pos_tensor
            v_cand_pos = v_cand*cand_pos_tensor

            #candidate_pos = self.word_graph_attention(candidate, hidden_states, k_cand_pos, v_cand_pos, "candidate_pos") #
            candidate_pos = self.word_graph_attention(candidate, last_token, k_cand_pos, v_cand_pos, "candidate_pos") #

            cand_pos_tensor = ((cand_pos_tensor-1)*(-1))
            v_cand_neg = v_cand*cand_pos_tensor
            #candidate_neg = self.word_graph_attention(candidate, hidden_states, None, v_cand, "candidate_neg") #
            #candidate_neg = self.word_graph_attention(candidate, hidden_states, None, v_cand_neg, "candidate_neg") #
            candidate_neg = self.word_graph_attention(candidate, last_token, None, v_cand_neg, "candidate_neg") #

            candidate = candidate_pos + candidate_neg

        sequence_output = encoded_layers[-1]
        ###
        sequence_output = sequence_output[input_ids == 2]
        ###
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        return encoded_layers, pooled_output, candidate
        #return sequence_output, pooled_output, candidate




class BertForPreTraining(PreTrainedBertModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    #def __init__(self, config):
    def __init__(self, config, args):
        super(BertForPreTraining, self).__init__(config)
        #self.bert = BertModel(config)
        self.bert = BertModel(config, args)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)
        self.args = args
        #self.word_graph_attention = Word_Graph_Attention(args.K_V_dim,args.Q_dim,args.graphsage,args.self_att)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
        input_ent=None, ent_mask=None, next_sentence_label=None, candidate=None, ent_labels=None, k=None, v=None, k_cand=None, v_cand=None, cand_pos_tensor=None):
        # the id in ent_labels should be consistent with the order of candidate.

        #sequence_output, pooled_output, hidden_states_6th_original = self.bert(input_ids, token_type_ids, attention_mask, input_ent, ent_mask, output_all_encoded_layers=False, k=k, v=v)

        sequence_output, pooled_output, candidate_emb = self.bert(input_ids, token_type_ids, attention_mask, input_ent, ent_mask, output_all_encoded_layers=False, k=k, v=v, candidate=candidate, k_cand=k_cand, v_cand=v_cand, cand_pos_tensor=cand_pos_tensor, mode="pretrain")



        prediction_scores, seq_relationship_score, prediction_scores_ent = self.cls(sequence_output, pooled_output, candidate_emb)

        #prediction_scores_ent -->return result

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_labels[next_sentence_label.view(-1) == 1, :] = -1
            ent_labels[next_sentence_label.view(-1) == 1, :] = -1
            ###
            #print("===")
            #print(prediction_scores.shape)
            #print(prediction_scores.view(-1,self.config.vocab_size).shape)
            #print(masked_lm_labels.view(-1).shape)
            #print("===")
            #exit()
            ###
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            #next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))


            #100 --> transE dim
            ent_ae_loss = loss_fct(prediction_scores_ent.view(-1, candidate.shape[1]), ent_labels.view(-1))

            #total_loss = masked_lm_loss + next_sentence_loss + ent_ae_loss
            total_loss = masked_lm_loss + ent_ae_loss
            #original_loss = masked_lm_loss + next_sentence_loss
            original_loss = masked_lm_loss
            return total_loss, original_loss
        else:
            return prediction_scores, seq_relationship_score, prediction_scores_ent


class BertForMaskedLM(PreTrainedBertModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, input_ents, ent_mask=None, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, input_ents, ent_mask,
                                       output_all_encoded_layers=False, k=k, v=v)
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores


class BertForNextSentencePrediction(PreTrainedBertModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        seq_relationship_score = self.cls( pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score



class BertForEntityTyping(PreTrainedBertModel):
    def __init__(self, config, num_labels=2, args=None):
        super(BertForEntityTyping, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.typing = nn.Linear(config.hidden_size, num_labels, False)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, labels=None, k=None, v=None):
        #_, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, input_ent, ent_mask, output_all_encoded_layers=False)
        encoded_layer, pooled_output, candidate_emb = self.bert(input_ids, token_type_ids, attention_mask, input_ent.long(), ent_mask, output_all_encoded_layers=False, k=v, v=v, mode="typing")

        #CLS: #ENT#
        pooled_output = self.dropout(pooled_output)
        logits = self.typing(pooled_output)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            #print(logits.dtype)
            #print(labels.dtype)
            #exit()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss, logits
        else:
            return logits


class BertForSTSB(PreTrainedBertModel):
    def __init__(self, config, num_labels=2):
        super(BertForSTSB, self).__init__(config)
        self.num_labels = 2
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

        self.m = torch.nn.LogSoftmax(-1)
        self.mm = torch.nn.Softmax(-1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, input_ent, ent_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        probs = self.m(logits)

        if labels is not None:
            #loss_fct = CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            per_example_loss = -torch.sum(labels * probs, -1)
            loss = torch.mean(per_example_loss)
            return loss
        else:
            return self.mm(logits)

#Fix here
class BertForSequenceClassification(PreTrainedBertModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels=2, args=None):
        super(BertForSequenceClassification, self).__init__(config)
        ##############
        #####GAT######
        #self.embed = embed
        #self.adj_list = adj_list
        ##############
        ##############
        self.num_labels = num_labels
        self.bert = BertModel(config,args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        ###
        #self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier = nn.Linear(config.hidden_size*2, num_labels)
        ###
        self.apply(self.init_bert_weights)

    #def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, labels=None):
    ############
    #def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, labels=None):
    #def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, labels=None, device=None, input_ent_emb=None):
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, labels=None, k=None, v=None):


        #print(k.dtype)
        #print(v.dtype)
        #print("------")

        #_, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, input_ent.long(), ent_mask, output_all_encoded_layers=False, subgraph=subgraph, subgraph_id_map_rep=subgraph_id_map_rep)
        #_, pooled_output, hidden_states_6th_original = self.bert(input_ids, token_type_ids, attention_mask, input_ent.long(), ent_mask, output_all_encoded_layers=False, input_ent_emb=input_ent_emb)
        seq_out, pooled_output, hidden_states_6th_original = self.bert(input_ids, token_type_ids, attention_mask, input_ent.long(), ent_mask, output_all_encoded_layers=False, k=k, v=v, candidate=None, k_cand=None, v_cand=None, cand_pos_tensor=None, mode="classfication")

        #_, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, input_ent, ent_mask, output_all_encoded_layers=False)

        ###tacred
        #head = seq_out[input_ids == 40117]
        #tail = seq_out[input_ids == 50210]
        ###

        ###fewrel
        head = seq_out[input_ids == 849]
        tail = seq_out[input_ids == 68]
        ###

        pooled_output = torch.cat([head,tail], -1)
        #pooled_output = self.dropout(pooled_output)
        #logits = self.classifier(pooled_output)
        ###
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            #print("===")
            #print(logits.shape)
            #print(logits.view(-1,self.num_labels).shape)
            #print("===")
            #print(labels.shape)
            #print(labels.view(-1).shape)
            #print("===")
            #print("===")
            #exit()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits



class BertForNQ(PreTrainedBertModel):

    def __init__(self, config, num_choices=2):
        super(BertForNQ, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, choice_mask=None, labels=None):
        #if choice_mask==None:
        #    _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, input_ent, ent_mask, output_all_encoded_layers=False)
        #    pooled_output = self.dropout(pooled_output)
        #    logits = self.classifier(pooled_output)
        #    return logits

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_input_ent = input_ent.view(-1, input_ent.size(-2), input_ent.size(-1))
        flat_ent_mask = ent_mask.view(-1, ent_mask.size(-1))
        _, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, flat_input_ent, flat_ent_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        null_socre = torch.zeros([labels.shape[0],1]).cuda()
        reshaped_logits = torch.cat([null_socre, reshaped_logits], -1) + choice_mask

        if labels is not None:
            weight = torch.FloatTensor([0.3]+[1]*16).cuda()
            loss_fct = CrossEntropyLoss(weight)
            loss = loss_fct(reshaped_logits, labels+1)
            return loss
        else:
            return reshaped_logits

class BertForMultipleChoice(PreTrainedBertModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
    input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
    token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_choices = 2

    model = BertForMultipleChoice(config, num_choices)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_choices=2):
        super(BertForMultipleChoice, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        _, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits


class BertForTokenClassification(PreTrainedBertModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels=2):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class BertForQuestionAnswering(PreTrainedBertModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: either
            - a BertConfig class instance with the configuration to build a new model, or
            - a str with the name of a pre-trained model to load selected in the list of:
                . `bert-base-uncased`
                . `bert-large-uncased`
                . `bert-base-cased`
                . `bert-base-multilingual`
                . `bert-base-chinese`
                The pre-trained model will be downloaded and cached if needed.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, start_positions=None, end_positions=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, input_ent, ent_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits
