import json
import torch
from model.bert import BertForTextClassification
from model.roberta import RobertaForTextClassification
from model.xlnet import XLNetForTextClassification

def load_txt(path):
    with open(path, 'r') as f:
        data = [x.replace('\n','') for x in f.readlines() if len(x)>0]
    return data

def save_txt(data, path):
    with open(path,'w') as f:
        for item in data:
            line = item.replace('\n', '') + '\n'
            f.write(line)

def load_json(path):
    with open(path, 'r') as handle:
        data = json.load(handle)
    return data

def save_json(data, path):
    with open(path, 'w') as handle:
        json.dump(data, handle)

def load_model(checkpoint, device, pretrained_model_path=None, model_name='xlnet', num_classes=2):
    if 'state' in checkpoint and pretrained_model_path != None:
        print('loading state model !!!!!')
        if model_name == 'bert':
            model = BertForTextClassification(pretrained_model_path, num_classes, device)
        elif model_name == 'xlnet':
            model = XLNetForTextClassification(pretrained_model_path, num_classes, device)
        elif model_name == 'roberta':
            model = RobertaForTextClassification(pretrained_model_path, num_classes, device)
        else:
            raise ValueError('model name wrong')
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    else:
        with open(checkpoint, 'rb') as f:
            model = torch.load(f, map_location=device)
        model.eval()
    return model