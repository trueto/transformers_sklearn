import torch
import torch.nn as nn
from transformers_sklearn.utils import FocalLoss
from transformers import BertPreTrainedModel, BertModel
from transformers_sklearn.model_albert_fix.modeling_albert import AlbertModel, AlbertPreTrainedModel


class BertForEL(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dense_1 = nn.Linear(config.hidden_size, 128)
        self.dropout = nn.Dropout(p=0.15)
        self.dense_2 = nn.Linear(128, self.num_labels)
        self.dense_3 = nn.Linear(self.num_labels, 1)


        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        start_positions=None,
        end_positions=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        # [b,1,dim]
        seq_len = sequence_output.shape[1]
        first_token_tensors = sequence_output[:, [0]*seq_len]
        start_entity_tensors = sequence_output[:, start_positions+1]
        end_entity_tensors = sequence_output[:, end_positions+1]

        merge_tensors = torch.cat([first_token_tensors,
                                   start_entity_tensors,
                                   end_entity_tensors], dim=1)
        # [b, dim]
        mean_tensors = merge_tensors.mean(dim=1)

        logits = self.dropout(self.dense_1(mean_tensors))
        logits = self.dense_2(logits)

        outputs = (logits,)
        if labels is not None:
            if self.num_labels == 1:
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits.view(-1), labels.view(-1))
            elif self.num_labels == 2:
                loss_fn = FocalLoss()
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        if self.num_labels == 2:
            match_score = self.dense_3(logits)
            outputs = outputs + (match_score,)

        return outputs

class AlbertForEL(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dense_1 = nn.Linear(config.hidden_size, 128)
        self.dropout = nn.Dropout(p=0.15)
        self.dense_2 = nn.Linear(128, self.num_labels)
        self.dense_3 = nn.Linear(self.num_labels, 1)


        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        start_positions=None,
        end_positions=None):

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        # [b,1,dim]
        seq_len = sequence_output.shape[1]
        first_token_tensors = sequence_output[:, [0]*seq_len]
        start_entity_tensors = sequence_output[:, start_positions+1]
        end_entity_tensors = sequence_output[:, end_positions+1]

        merge_tensors = torch.cat([first_token_tensors,
                                   start_entity_tensors,
                                   end_entity_tensors], dim=1)
        # [b, dim]
        mean_tensors = merge_tensors.mean(dim=1)

        logits = self.dropout(self.dense_1(mean_tensors))
        logits = self.dense_2(logits)

        outputs = (logits,)
        if labels is not None:
            if self.num_labels == 1:
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits.view(-1), labels.view(-1))
            elif self.num_labels == 2:
                loss_fn = FocalLoss()
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        match_score = self.dense_3(logits)
        outputs = outputs + (match_score,)

        return outputs