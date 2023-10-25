# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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


class BCQConfigForWebshop(PretrainedConfig):
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



class BCQModelForWebshop(PreTrainedModel):
    """This class implements the encoding model for the BCQ algorithm.

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

    config_class = BCQConfigForWebshop

    def __init__(self, config):
        super().__init__(config)
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        if config.pretrained_bert:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.bert = BertModel(config)
        self.bert.resize_token_embeddings(30526)
        self.attn = BiAttention(768, 0.0)
        
        self.q_linear_1 = nn.Linear(768 * 4, 768)
        self.q_relu = nn.ReLU()
        self.q_linear_2 = nn.Linear(768, 1)
        
        self.i_linear_1 = nn.Linear(768 * 4, 768)
        self.i_relu = nn.ReLU()
        self.i_linear_2 = nn.Linear(768, 1)
        
        if config.image:
            self.image_linear = nn.Linear(512, 768)
        else:
            self.image_linear = None

    def forward(self, state_input_ids, state_attention_mask, action_input_ids, action_attention_mask, sizes, images=None, labels=None):
        sizes = sizes.tolist()

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
        
        # q values
        q_state_action_rep = self.q_relu(self.q_linear_1(state_action_rep))
        q_act_values = get_aggregated(q_state_action_rep, act_lens, 'mean')
        q_act_values = self.q_linear_2(q_act_values).squeeze(1)
        q_logits = [act for act in q_act_values.split(sizes)]
        
        # i values
        i_state_action_rep = self.i_relu(self.i_linear_1(state_action_rep))
        i_act_values = get_aggregated(i_state_action_rep, act_lens, 'mean')
        i_act_values = self.i_linear_2(i_act_values).squeeze(1)
        i_logits = [act for act in i_act_values.split(sizes)]
        i_log_softmax = [F.log_softmax(act, dim=-1) for act in i_act_values.split(sizes)]
        
        
        return q_logits, i_log_softmax, i_logits
    
