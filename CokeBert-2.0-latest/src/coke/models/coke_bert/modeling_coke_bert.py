import copy
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import load_tf_weights_in_bert
from transformers.activations import ACT2FN
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import (
    PreTrainedModel,
    BertEmbeddings,
    BertAttention,
    BertLayer,
    BertEncoder,
    BertOutput,
    BertPooler,
    BertLMPredictionHead,
    BertPredictionHeadTransform
)
from .configuration_coke_bert import CokeBertConfig


class CokeBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_ent = nn.Linear(config.hidden_size_ent, config.intermediate_size, bias=False)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states_text, hidden_states_ent):
        hidden_states_text = self.dense(hidden_states_text)
        hidden_states_ent = self.dense_ent(hidden_states_ent)

        hidden_states = hidden_states_text + hidden_states_ent
        hidden_states = self.intermediate_act_fn(hidden_states)
        
        return hidden_states

class CokeBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_text = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense_ent = nn.Linear(config.intermediate_size, config.hidden_size_ent)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm_ent = BertLayerNorm(config.hidden_size_ent, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_hidden_states, input_tensor_text, input_tensor_ent):
        hidden_states_text = self.dense(input_hidden_states)
        hidden_states_text = self.dropout(hidden_states_text)
        hidden_states_text = self.LayerNorm(hidden_states_text + input_tensor_text)

        hidden_states_ent = self.dense_ent(input_hidden_states)
        hidden_states_ent = self.dropout(hidden_states_ent)
        hidden_states_ent = self.LayerNorm_ent(hidden_states_ent + input_tensor_ent)

        return hidden_states_text, hidden_states_ent

class DynamicKnowledgeContextEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Text
        self.attention = BertAttention(config)
        self.output_text = BertOutput(config)

        self.intermediate = CokeBertIntermediate(config)

        # Entity
        config_ent = copy.deepcopy(config)
        config_ent.hidden_size = config.hidden_size_ent
        config_ent.num_attention_heads = config.num_attention_heads_ent
        self.output_ent = BertOutput(config_ent)

    def forward(self, hidden_states, attention_mask, hidden_states_ent, attention_mask_ent, ent_mask):
        attention_output_text = self.attention(hidden_states, attention_mask)[0]
        attention_output_ent = hidden_states_ent * ent_mask
        
        intermediate_output = self.intermediate(attention_output_text, attention_output_ent)

        layer_output_text = self.output_text(intermediate_output, attention_output_text)
        layer_output_ent = self.output_ent(intermediate_output, attention_output_ent)

        return layer_output_text, layer_output_ent
        
class KnowledgeFusionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Text
        self.attention_text = BertAttention(config)
        self.output_text = BertOutput(config)

        self.intermediate = CokeBertIntermediate(config)
        
        # Entity
        config_ent = copy.deepcopy(config)
        config_ent.hidden_size = config.hidden_size_ent
        config_ent.num_attention_heads = config.num_attention_heads_ent
        self.attention_ent = BertAttention(config_ent)
        self.output_ent = BertOutput(config_ent)
    
    def forward(self, hidden_states, attention_mask, hidden_states_ent, attention_mask_ent, ent_mask):
        attention_output_text = self.attention_text(hidden_states, attention_mask)[0]
        attention_output_ent = self.attention_ent(hidden_states_ent, attention_mask_ent)[0]

        attention_output_ent = attention_output_ent * ent_mask

        intermediate_output = self.intermediate(attention_output_text, attention_output_ent)
        
        layer_output_text = self.output_text(intermediate_output, attention_output_text)
        layer_output_ent = self.output_ent(intermediate_output, attention_output_ent)

        return layer_output_text, layer_output_ent

class DynamicKnowledgeContextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([DynamicKnowledgeContextEncoderLayer(config) for l in config.layer_types if l == 'knowledge_enc'])

    def forward(
        self, 
        hidden_states_text, 
        attention_mask_text,
        hidden_states_ent, 
        attention_mask_ent, 
        ent_mask,
        layer_head_mask=None, 
        encoder_hidden_states=None, 
        encoder_attention_mask=None, 
        past_key_value=None, 
        output_attentions=None
    ):
        for i, layer_module in enumerate(self.layer):
            hidden_states_text, hidden_states_ent = layer_module(
                hidden_states_text,
                attention_mask_text,
                hidden_states_ent,
                attention_mask_ent,
                ent_mask
            )

        return tuple([hidden_states_text, hidden_states_ent])

class KnowledgeFusionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([KnowledgeFusionLayer(config) for l in config.layer_types if l == 'knowledge_fusion'])

    def forward(
        self, 
        hidden_states_text, 
        attention_mask_text,
        hidden_states_ent, 
        attention_mask_ent, 
        ent_mask,
        layer_head_mask=None, 
        encoder_hidden_states=None, 
        encoder_attention_mask=None, 
        past_key_value=None, 
        output_attentions=None
    ):
        for i, layer_module in enumerate(self.layer):
            hidden_states_text, hidden_states_ent = layer_module(
                hidden_states_text,
                attention_mask_text,
                hidden_states_ent,
                attention_mask_ent,
                ent_mask
            )

        return tuple([hidden_states_text, hidden_states_ent])

class CokeBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CokeBertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "coke_bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value

class CokeBertModel(CokeBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.embeddings = BertEmbeddings(config)

        config_text = copy.deepcopy(config)
        config_text.num_hidden_layers = len([i for i in config.layer_types if i == 'text_enc'])
        self.text_encoder = BertEncoder(config_text)

        self.dynamic_knowledge_context_encoder = DynamicKnowledgeContextEncoder(config)
        self.knowledge_fusion_encoder = KnowledgeFusionEncoder(config)
        self.word_graph_attention = WordGraphAttention(config)
        self.pooler = BertPooler(config)

        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        input_ent=None,
        ent_mask=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        all_encoder_layers=None,
        output_all_encoded_layers=True,
        k_1=None, 
        v_1=None, 
        k_2=None, 
        v_2=None, 
    ):
        ## Copied from BertModel
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.text_encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        hidden_states_text = encoder_outputs[0] # For word graph attention

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

        ent_mask = ent_mask.to(dtype=next(self.parameters()).dtype).unsqueeze(-1)

        if len(input_ent[input_ent!=0]) == 0:
            hidden_states_ent = torch.zeros(input_ent.shape[0], input_ent.shape[1],200).cuda().half()
        else:
            hidden_states_ent = self.word_graph_attention(input_ent, hidden_states, k_1, v_1, k_2, v_2, "entity")
        
        hidden_states, hidden_states_ent = self.dynamic_knowledge_context_encoder(
            hidden_states, 
            extended_attention_mask, 
            hidden_states_ent, 
            extended_ent_mask, 
            ent_mask
        )
        sequence_output, hidden_states_ent = self.knowledge_fusion_encoder(
            hidden_states, 
            extended_attention_mask, 
            hidden_states_ent, 
            extended_ent_mask, 
            ent_mask
        )

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return sequence_output, pooled_output, hidden_states_text
        # return BaseModelOutputWithPoolingAndCrossAttentions(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     past_key_values=encoder_outputs.past_key_values,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        #     cross_attentions=encoder_outputs.cross_attentions,
        # )

class WordGraphAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.K_V_linear_1 = nn.Linear(config.K_V_dim, config.K_V_dim, bias=False)
        self.K_V_linear_2 = nn.Linear(config.K_V_dim, config.K_V_dim, bias=False)
        self.V_linear_1 = nn.Linear(config.K_V_dim, config.K_V_dim, bias=False)
        self.V_linear_2 = nn.Linear(config.K_V_dim, config.K_V_dim, bias=False)
        self.Q_linear_1 = nn.Linear(config.Q_dim, config.K_V_dim, bias=True)
        self.Q_linear_2 = nn.Linear(config.Q_dim, config.K_V_dim, bias=True)
        self.comb = nn.Linear(config.K_V_dim*2, config.K_V_dim, bias=False)

        self.softmax_dim_2 = nn.Softmax(dim=2)
        self.softmax_dim_3 = nn.Softmax(dim=3)
        self.LeakyReLU = nn.LeakyReLU()
        self.Tanh = nn.Tanh()

        if self.config.graphsage:
            self.graphsage_linear = nn.Linear(config.K_V_dim * 2, config.K_V_dim, bias=False)

        self.init_weight()

    def init_weight(self):
        for module in [self.K_V_linear_1, self.K_V_linear_2, self.V_linear_1, self.V_linear_2, self.Q_linear_1, self.Q_linear_2, self.comb]:
            init.xavier_uniform_(module.weight)

        if self.config.graphsage:
            init.xavier_uniform_(self.graphsage_linear.weight)

    def self_attention(self, Q, K_1, V_1, K_2, V_2, mode=None):
        Q_2 = self.Q_linear_2(Q)
        Q_2 = self.Tanh(Q_2)
        Q_2 = Q_2.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        if self.config.neighbor_hop == 2:
            K = self.K_V_linear_2(K_2)
            attention = ((Q_2*K).sum(4)).div(math.sqrt(self.config.K_V_dim))
            attention = attention.masked_fill(attention==0, float('-10000'))
            attention = self.softmax_dim_3(self.LeakyReLU(attention))
            attention = attention.masked_fill(attention==float(1/attention.shape[-1]), float(0)) # don't need to
            attention = attention.unsqueeze(3)
            sentence_entity_reps = attention.matmul(V_2).squeeze(3)

            if mode == "candidate_pos":
                V_mask = V_1.sum(3)
                V_mask = V_mask.masked_fill(V_mask!=0, float(1)).unsqueeze(-1)
                sentence_entity_reps = V_mask*sentence_entity_reps
                V_1 = V_mask*V_1
                V_1 = torch.cat([V_1, sentence_entity_reps],-1)

            elif mode == "candidate_neg":
                V_mask = V_1.sum(3)
                V_mask = ( V_mask.masked_fill(V_mask!=0, float(1)).unsqueeze(-1)-1 ) * -1
                sentence_entity_reps = V_mask*sentence_entity_reps
                V_1 = V_mask*V_1
                V_1 = torch.cat([V_1, sentence_entity_reps],-1)
            else:
                V_1 = torch.cat([V_1, sentence_entity_reps],-1)

            Q_1 = self.Q_linear_1(Q)
            Q_1 = self.Tanh(Q_1)
            Q_1 = Q_1.unsqueeze(1).unsqueeze(2)
            K = self.K_V_linear_1(K_1)
            attention = ((Q_1*K).sum(3)).div(math.sqrt(self.config.K_V_dim))
            attention = attention.masked_fill(attention==0, float('-10000'))
            attention = self.softmax_dim_2(self.LeakyReLU(attention))
            attention = attention.masked_fill(attention==float(1/attention.shape[-1]), float(0)) # don't need to
            attention = attention.unsqueeze(2)
            sentence_entity_reps = attention.matmul(V_1).squeeze(2)

            return sentence_entity_reps
        
        elif self.config.neighbor_hop == 1:
            if mode == "candidate_pos":
                V_mask = V_1.sum(3)
                V_mask = V_mask.masked_fill(V_mask!=0, float(1)).unsqueeze(-1)
                V_1 = V_mask*V_1

            elif mode == "candidate_neg":
                V_mask = V_1.sum(3)
                V_mask = ( V_mask.masked_fill(V_mask!=0, float(1)).unsqueeze(-1)-1 ) * -1
                V_1 = V_mask*V_1
            
            else:
                V_1 = V_1

            Q_1 = self.Q_linear_1(Q)
            Q_1 = self.Tanh(Q_1)
            Q_1 = Q_1.unsqueeze(1).unsqueeze(2)
            K = self.K_V_linear_1(K_1)
            attention = ((Q_1*K).sum(3)).div(math.sqrt(self.config.K_V_dim))
            attention = attention.masked_fill(attention==0, float('-10000'))
            attention = self.softmax_dim_2(self.LeakyReLU(attention))
            attention = attention.masked_fill(attention==float(1/attention.shape[-1]), float(0)) # don't need to
            attention = attention.unsqueeze(2)
            sentence_entity_reps = attention.matmul(V_1).squeeze(2)

            return sentence_entity_reps
        
        else:
            raise NotImplementedError

    def forward(self, input_ent, q, k_1, v_1, k_2, v_2, mode):

        if mode == "entity":
            q = q[:,0,:] # all input: 0, !=0
            combined = self.self_attention(q, k_1, v_1, k_2, v_2)
            hidden_states_ent = torch.zeros(input_ent.shape[0], input_ent.shape[1], self.config.K_V_dim * self.config.neighbor_hop).half().cuda()
            ent_pos_s = torch.nonzero(input_ent) # id start from 0

            for batch in range(input_ent.shape[0]):
                for i,index in enumerate(ent_pos_s[ent_pos_s[:,0]==batch]):
                    hidden_states_ent[batch][int(index[1])] = combined[batch][i]

        elif mode == "candidate_neg":
            q = q[:,0,:] # all input: 0, !=0
            hidden_states_ent = self.self_attention(q, k_1, v_1, k_2, v_2, "candidate_neg") ###

        elif mode == "candidate_pos":
            q = q[:,0,:] # all input: 0, !=0
            hidden_states_ent = self.self_attention(q, k_1, v_1, k_2, v_2, "candidate_pos")

        else:
            raise NotImplementedError("Graph attention mode Wrong!!")

        return hidden_states_ent

class CokeBertEntPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size_ent)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size_ent, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class CokeBertEntPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = CokeBertEntPredictionHeadTransform(config)

    def forward(self, hidden_states, candidate_emb):
        hidden_states = self.transform(hidden_states)

        return torch.matmul(hidden_states, candidate_emb.transpose(1,2)) # will return torch.Size([2, 256, 9])

class CokeBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.predictions_ent = CokeBertEntPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output, candidate_emb):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        prediction_scores_ent = self.predictions_ent(sequence_output, candidate_emb)
        return prediction_scores, seq_relationship_score, prediction_scores_ent

class CokeBertForPreTraining(CokeBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.cokebert = CokeBertModel(config)
        self.cls = CokeBertPreTrainingHeads(config)
        
        self.init_weights()

    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        token_type_ids=None, 
        labels=None,
        input_ent=None, 
        ent_mask=None, 
        next_sentence_label=None, 
        candidate=None, 
        ent_labels=None, 
        k_1=None, 
        v_1=None, 
        k_2=None, 
        v_2=None, 
        k_cand_1=None, 
        v_cand_1=None, 
        k_cand_2=None, 
        v_cand_2=None, 
        cand_pos_tensor=None
    ):
        
        sequence_output, pooled_output, hidden_states_text = self.cokebert(
            input_ids, 
            attention_mask, 
            token_type_ids, 
            input_ent=input_ent,
            ent_mask=ent_mask,
            output_all_encoded_layers=False, 
            k_1=k_1,
            v_1=v_1,
            k_2=k_2,
            v_2=v_2
        )

        # Denoised Entity Alignment
        k_cand_1 = torch.cat(hidden_states_text.shape[0]*[k_cand_1])
        v_cand_1 = torch.cat(hidden_states_text.shape[0]*[v_cand_1])
        cand_pos_tensor = cand_pos_tensor.float().half().unsqueeze(2).unsqueeze(3)
        v_cand_pos_1 = v_cand_1*cand_pos_tensor

        candidate_pos = self.cokebert.word_graph_attention(candidate, hidden_states_text, k_cand_1, v_cand_pos_1, k_cand_2, v_cand_2, "candidate_pos")
        cand_pos_tensor = ((cand_pos_tensor-1)*(-1))
        v_cand_neg_1 = v_cand_1 * cand_pos_tensor
        candidate_neg = self.cokebert.word_graph_attention(candidate, hidden_states_text, k_cand_1, v_cand_neg_1, k_cand_2, v_cand_2, "candidate_neg")
        candidate_emb = candidate_pos + candidate_neg

        prediction_scores, seq_relationship_score, prediction_scores_ent = self.cls(sequence_output, pooled_output, candidate_emb)

        if labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            ent_ae_loss = loss_fct(prediction_scores_ent.view(-1, candidate.shape[1]), ent_labels.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss + ent_ae_loss
            original_loss = masked_lm_loss + next_sentence_loss
            return total_loss, original_loss
        else:
            return prediction_scores, seq_relationship_score, prediction_scores_ent


class CokeBertForRelationClassification(CokeBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.cokebert = CokeBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size*2)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        token_type_ids=None, 
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None, 
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        input_ent=None, 
        ent_mask=None, 
        k_1=None, 
        v_1=None, 
        k_2=None, 
        v_2=None
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.cokebert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            input_ent=input_ent.long(), 
            ent_mask=ent_mask, 
            k_1=k_1, 
            v_1=v_1, 
            k_2=k_2, 
            v_2=v_2
        )

        seq_output = outputs[0]
        head = seq_output[input_ids == 1601]
        tail = seq_output[input_ids == 1089]
        pooled_output = torch.cat([head, tail], -1)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )
