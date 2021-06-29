import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLNetConfig, XLNetTokenizer, XLNetForSequenceClassification

class XLNetForTextClassification(nn.Module):
    def __init__(self, pretrained_model_path, num_classes, device, d_model=1024, n_layer=24, n_head=16, \
            d_inner=4096, ff_activation='gelu', untie_r=True, attn_type='bi',initializer_range=0.02, \
            layer_norm_eps=1e-12, dropout=0.1):
        super(XLNetForTextClassification, self).__init__()

        print('Reloading pretrained models...')
        self.tokenizer = XLNetTokenizer.from_pretrained(pretrained_model_path)
        self.tokenizer.model_max_length = 512
        self.model = XLNetForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=num_classes).to(device)
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
        return proba, logits, hidden_states, attentions