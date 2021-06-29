import numpy as np
import os
import random
import torch
from utils import load_model
import argparse
import warnings
warnings.filterwarnings("ignore")

# greedy select a token and replace it with another token (sample from synonyms or vocab)
# return candidate which achieves max probability shift
def gsgr(model, text, y_truth, pad_token, synonyms, vocab_lst, device, bs, n):
    if(len(text)<10):
        return None, None
    candidate_lst = []
    idx_lst = []
    while(len(candidate_lst)<bs):
        k = np.random.randint(low=1, high=3)
        sampled_idx = random.sample(list(range(len(text))), k)
        tmp_text = text
        for idx in sampled_idx:
            tmp_text = tmp_text[:idx] + pad_token + tmp_text[idx+1:]
        candidate_lst.append(tmp_text)
        idx_lst.append(sampled_idx)
    candidate_lst = list(set(candidate_lst))
    proba_org = model([text], device)[0].cpu().detach().numpy()
    proba_candidate = model(candidate_lst, device)[0].cpu().detach().numpy()

    
    if(y_truth==0):
        diff = proba_candidate-proba_org
    elif(y_truth==1):
        diff = proba_org-proba_candidate

    # find position wrt max probability shift
    #idx = np.argmax(diff[:,1])
    weak_spot_idx_lst = list(np.argsort(diff[:, 1])[:n])
    result_text_lst, result_idx_lst = [], []
    for idx in weak_spot_idx_lst:
        for i in idx_lst[idx]:
            w = text[i]
            if(w in synonyms.keys()):
                new_w = random.choice(synonyms[w])
            else:
                new_w = random.choice(vocab_lst)
            noisy_text = text[:i] + new_w + text[i+1:]
        result_text_lst.append(noisy_text)
        result_idx_lst.append(idx_lst[idx])
    return result_text_lst, result_idx_lst


def create_adv_data(checkpoint_path, gsgr, data_lst, synonyms, vocab_lst, device, bs=50, n=1):
    model = load_model(checkpoint_path, device)
    result_lst = []

    pad_token = '[pad]'
    for idx, item in enumerate(data_lst):
        text = item['text']
        label = item['label']
        text_adv_lst = [text]
        try:
            if(len(list(text))>10 and len(list(text))<100):
                text_adv_lst, _ = gsgr(model, text, label, pad_token, synonyms, vocab_lst, device=device, bs=bs, n=n)
        except:
            pass
        for text_adv in text_adv_lst:
            example = {'text' : text_adv, 'label' : label, 'idx' : idx}
            result_lst.append(example)
        if(len(result_lst)%2000==0):
            print('done {}/{}'.format(idx, len(data_lst)), flush=True)
    return result_lst



if __name__=='__main__':
    from utils import *

    parser = argparse.ArgumentParser()
    parser.add_argument("--which_data", type=str, default='metaphor')
    parser.add_argument("--checkpoint", type=str, default='')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--synonyms_path", type=str, default='synonym.json')
    parser.add_argument("--vocab_path", type=str, default='3500_frequent_chars.txt')
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument('--num_samples_per_attack', type=int, default=1)
    parser.add_argument('--attacked_data_save_path', type=str, default='../../data/{}/data')


    args = parser.parse_args()

    which_data = args.which_data
    checkpoint = args.checkpoint
    synonyms_path = args.synonyms_path
    vocab_path = args.vocab_path
    batch_size = args.batch_size
    num_samples_per_attack = args.num_samples_per_attack
    device = args.device
    attacked_data_save_path = args.attacked_data_save_path

    synonyms = load_json(synonyms_path)
    vocab_lst = load_txt(vocab_path)

    print('attacking training data, please wait...')
    attacked_data_save_path = attacked_data_save_path.format(which_data)
    os.makedirs(attacked_data_save_path, exist_ok=True)


    data_lst = load_json('../../data/{}/train.json'.format(which_data))['data']

    attacked_data = create_adv_data(checkpoint, gsgr, data_lst, synonyms, vocab_lst, device, bs=batch_size, n=num_samples_per_attack)

    save_json(attacked_data, os.path.join(attacked_data_save_path, 'train_large.json'))
    print('finish attacks..')