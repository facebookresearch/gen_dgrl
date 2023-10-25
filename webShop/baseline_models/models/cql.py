import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertModel,
    BertConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from .modules import EncoderRNN, BiAttention, get_aggregated


class CQLConfigForWebshop(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        pretrained_bert=True,
        image=False,

        **kwargs
    ):
        self.pretrained_bert = pretrained_bert
        self.image = image
        super().__init__(**kwargs)



class QModelForWebshop(PreTrainedModel):
    """This class implements the Q-model for the CQL algorithm.

    Args:
        state_input_ids (torch.Tensor): Tokenized state input.
        state_attention_mask (torch.Tensor): Attention mask for state input.
        action_input_ids (torch.Tensor): Tokenized action input.
        action_attention_mask (torch.Tensor): Attention mask for action input.
        labels (torch.Tensor): action label for querying the Q-value.
        sizes (list): Number of actions for each state.

    Returns:
        q_vals: Q-values for corresponding actions in the current state.
    """

    config_class = CQLConfigForWebshop

    def __init__(self, config):
        super().__init__(config)
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        if config.pretrained_bert:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.bert = BertModel(config)
        self.bert.resize_token_embeddings(30526)
        self.attn = BiAttention(768, 0.0)
        self.linear_1 = nn.Linear(768 * 4, 768)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(768, 1)
        if config.image:
            self.image_linear = nn.Linear(512, 768)
        else:
            self.image_linear = None

    def forward(self, state_input_ids, state_attention_mask, action_input_ids, action_attention_mask, sizes, images=None, labels=None):
        sizes = sizes.tolist()
        # print(state_input_ids.shape, action_input_ids.shape)
        state_rep = self.bert(state_input_ids, attention_mask=state_attention_mask)[0]
        if images is not None and self.image_linear is not None:
            images = self.image_linear(images)
            state_rep = torch.cat([images.unsqueeze(1), state_rep], dim=1)
            state_attention_mask = torch.cat([state_attention_mask[:, :1], state_attention_mask], dim=1)
        action_rep = self.bert(action_input_ids, attention_mask=action_attention_mask)[0]
        state_rep = torch.cat([state_rep[i:i+1].repeat(j, 1, 1) for i, j in enumerate(sizes)], dim=0)
        state_attention_mask = torch.cat([state_attention_mask[i:i+1].repeat(j, 1) for i, j in enumerate(sizes)], dim=0)
        act_lens = action_attention_mask.sum(1).tolist()
        state_action_rep = self.attn(action_rep, state_rep, state_attention_mask)
        state_action_rep = self.relu(self.linear_1(state_action_rep))
        act_values = get_aggregated(state_action_rep, act_lens, 'mean')
        act_values = self.linear_2(act_values).squeeze(1)

        logits = [act for act in act_values.split(sizes)]
        if labels is None:
            return logits

        q_val = [logit[label] for logit, label in zip(logits, labels)]
            
        q_vals = torch.stack(q_val)
        
        return q_vals
