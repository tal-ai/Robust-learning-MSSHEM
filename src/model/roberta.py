import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


class RobertaForTextClassification(nn.Module):
    def __init__(self, pretrained_model_path, num_classes, device, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, \
            intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, 
            max_position_embeddings=512):
        super(RobertaForTextClassification, self).__init__()

        print('Reloading pretrained models...')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        self.tokenizer.model_max_length = 512
        self.model = RobertaForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=num_classes).to(device)
        self.softmax = torch.nn.Softmax(dim=1)
        self.device = device

    def forward(self, text_lst, device=None):
        result = self.tokenizer(text_lst, padding=True, return_tensors='pt', truncation=True)
        if(device==None):
            input_ids = result.input_ids.to(self.device)
            token_type_ids = result.token_type_ids.to(self.device)
            attention_mask =result.attention_mask.to(self.device)
        else:
            input_ids = result.input_ids.to(device)
            token_type_ids = result.token_type_ids.to(device)
            attention_mask =result.attention_mask.to(device)
        out = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, \
                                                        output_hidden_states=True, return_dict=True)
        logits = out.logits
        attentions = out.attentions
        hidden_states = out.hidden_states # one for the output of the embeddings + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).
        proba = self.softmax(logits)
        # return proba, logits, hidden_states, attentions
        return proba, logits, hidden_states