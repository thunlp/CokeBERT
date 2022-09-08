""" CokeBert model configuration """

from collections import OrderedDict
from typing import Mapping

from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class CokeBertConfig(PretrainedConfig):
    model_type = "coke_bert"

    def __init__(
        self,
        vocab_size=30522,
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
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        K_V_dim=100,
        neighbor_hop=2,
        Q_dim=768,
        graphsage=False,
        self_att=True,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
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
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout

        # Coke
        self.hidden_size_ent = K_V_dim * neighbor_hop
        self.K_V_dim = K_V_dim
        self.neighbor_hop = neighbor_hop
        self.Q_dim = Q_dim
        self.graphsage = False
        self.self_att = True


class CokeBertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
                ("token_type_ids", {0: "batch", 1: "sequence"}),
            ]
        )
