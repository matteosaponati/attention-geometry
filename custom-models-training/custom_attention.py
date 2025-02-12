from torch import nn
from transformers.models.bert.modeling_bert import BertSelfAttention


class BertSelfAttentionSymmetricInit(BertSelfAttention):
    """
    Custom Self Attention layer for BERT with symmetric initialization
    """

    def __init__(self, config):
        super().__init__(config)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.query.weight = nn.Parameter(self.key.weight.detach().clone())
        # to_symmetric_asymmetric_fast(self.key, self.query, self.num_attention_heads, self.attention_head_size, self.all_head_size)
