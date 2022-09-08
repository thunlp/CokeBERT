from modeling_comb import BertForSequenceClassification as BertForSequenceClassification_att
from modeling_ernie import BertForSequenceClassification as BertForSequenceClassification_ernie
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE


class Model(args, num_labels, ):

    def __init__(self, config):
        super(Model, self).__init__(config)
        self.model_att, _ = BertForSequenceClassification_att.from_pretrained(args.ernie_model,
              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
              num_labels = num_labels, args=args)

        self.model_ernie, _ = BertForSequenceClassification_ernie.from_pretrained("../"+args.ernie_model,
              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank),
              num_labels = num_labels)

    def forward(input_ids, segment_ids, input_mask, input_ent_neighbor_emb, ent_mask, label_ids):
        logits_ernie, loss_ernie = self.model_ernie(input_ids, segment_ids, input_mask, input_ent_neighbor_emb, ent_mask, label_ids)
        loss = self.model_att(input_ids, segment_ids, input_mask, input_ent, ent_mask, label_ids, k, v, logits_ernie)

        return logits_ernie, loss_ernie, loss
