#!/usr/bin/env python
# coding: utf-8
from utils import *
import pandas as pd
import os, re
from bpe import *

def write_list_2_txt(lst, save_path):
    with open(save_path, 'w') as f:
        for item in lst:
            item = item.replace('\n', '') + '\n'
            f.write(item)

def create_tr_val_data(bpe_model_path, save_dir, clean_lst, ocr_lst):
    bpe_model = reload_bpe_model(bpe_model_path)
    ocr_bpe_lst = apply_bpe_on_raw_text(bpe_model, ocr_lst)
    clean_bpe_lst = apply_bpe_on_raw_text(bpe_model, clean_lst)
    N = len(ocr_bpe_lst)
    print('all data count = {}'.format(N))
    ocr_bpe_lst_tr, ocr_bpe_lst_val = ocr_bpe_lst[:int(N*0.92)], ocr_bpe_lst[int(N*0.92):]
    clean_bpe_lst_tr, clean_bpe_lst_val = clean_bpe_lst[:int(N*0.92)], clean_bpe_lst[int(N*0.92):]
    print('data ready ......')
    write_list_2_txt(ocr_bpe_lst_tr, save_dir+'/train.noise')
    write_list_2_txt(ocr_bpe_lst_val, save_dir+'/valid.noise')
    write_list_2_txt(clean_bpe_lst_tr, save_dir+'/train.clean')
    write_list_2_txt(clean_bpe_lst_val, save_dir+'/valid.clean')
    print('data created ......')
    

def apply_bpe_on_raw_text(bpe_model, text_lst):
    result = []
    for text in text_lst:
        text_cut = ' '.join(bpe_model.EncodeAsPieces(str(text)))
        result.append(text_cut)
    return result


def main_data():
    bpe_model_path = './model/ch_bpe.model'
    save_dir = './data/fairseq'
    df = pd.read_csv('./data/ocr_clean.csv')
    clean_lst = df['clean_text'].tolist()
    ocr_lst = df['ocr_text'].tolist()
    create_tr_val_data(bpe_model_path, save_dir, clean_lst, ocr_lst)


if __name__ == "__main__":
    main_data()


# # create fairseq train data，source: clean text，target: noise text
# fairseq-preprocess --source-lang clean --target-lang noise \
#     --trainpref /share/作文批改/data/ocr适配/data/v03/train \
#     --validpref /share/作文批改/data/ocr适配/data/v03/valid \
#     --destdir /share/作文批改/data/ocr适配/噪声生成器/v03/bin-1 \
#     --workers=4



