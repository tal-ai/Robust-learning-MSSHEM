import fairseq_model_client
import logging
import re
import os
import unicodedata
import json
import sentencepiece as spm
import pandas as pd
from sklearn.utils import shuffle
import jieba
from Levenshtein import *


def is_repeat(text, thres=3):
    word_list = list(text)
    repeat_count = 0
    for i in range(0, len(word_list)-1):
        for j in range(i+1, len(word_list)):
            if word_list[i] == word_list[j]:
                repeat_count += 1
                if repeat_count > thres:
                    return True
            else:
                if repeat_count > thres:
                    return True
                repeat_count = 0
                break
    word_list = list(jieba.cut(text))
    repeat_count = 0

    for i in range(0, len(word_list)-1):
        for j in range(i+1, len(word_list)):
            if word_list[i] == word_list[j]:
                repeat_count += 1
                if repeat_count > thres:
                    return True
            else:
                if repeat_count > thres:
                    return True
                repeat_count = 0
                break
    return False

def load_json(path):
    with open(path, 'r') as handle:
        data = json.load(handle)
    return data

def save_json(data, path):
    with open(path, 'w') as handle:
        json.dump(data, handle)

def union_symbol(s):
    dic = {
        ',':'，', 
        ';':'；',
        '!':'！',
        '?':'？',
        ':':'：',
        '[' : '【',
        ']' : '】',
        '(' : '（',
        ')' : '）'
    } 
    for d, d_ in dic.items():
        s = s.replace(d, d_)
    return s

def is_match_by_distance(origin_sent, cand_sent):
    dis = distance(origin_sent, cand_sent)
    if dis / len(origin_sent) <= 0.2:
        return True
    return False

class SentencePieceBpe:
    def __init__(self, code_file):
        self.model = self.load_model(code_file)

    def load_model(self, code_file):
        s = spm.SentencePieceProcessor()
        s.Load(code_file)
        return s

    def segment(self, sentence):
        return " ".join(self.model.encode_as_pieces(sentence))

    def decode(self, sentence):
        return self.model.decode_pieces(sentence.split(" "))


class Denoiser():
    def __init__(self, config):
        current_dir_path = current_dir_path = os.path.dirname(os.path.realpath(__file__))

        checkpoint_path = os.path.join(current_dir_path, config.get('nmt_model_path'))
        data_path = os.path.join(current_dir_path, config.get('dictionary_path'))
        bpe_code_file = os.path.join(current_dir_path, config.get('bpe_model_path'))
        beam = config.get('beam_search_width')

        if(config.get('nbest')):
            self.nbest = config.get('nbest')
        else:
            self.nbest= 1
            
        model_params = {"--path":checkpoint_path,
                        "--beam":str(beam),
                        "--nbest":str(self.nbest)
                        }

        logging.info('Initializing bpe model...')
        self.bpe = SentencePieceBpe(bpe_code_file)
        
        logging.info('Initializing transformer...')
        self.model = fairseq_model_client.get_model(data_path,**model_params)

        logging.info('All model initialization complete...')

    def cut_chinese(self, text, max_len):
        if len(text) <= max_len:
            return [text]
        text_list = []
        # 先split再拼接
        temp = re.sub(r'(\!|\?|。”|！”|？”|？|！|。|；|……|，)', r'\1@#split#@', text)
        all_segs = re.split(r'@#split#@', temp)
        cur_str = all_segs[0]
        i = 1
        while i < len(all_segs):
            if len(cur_str)+len(all_segs[i]) <= max_len:
                cur_str += all_segs[i]
            else:
                text_list.append(cur_str)
                cur_str = all_segs[i]
            i += 1
        text_list.append(cur_str)
        return [x for x in text_list if len(x)>0]

    def cut_long_sentence(self, text):
        ch_max_len = 60
        return self.cut_chinese(text, ch_max_len)
            
    def nmt_model_predict(self, sentence):
        if type(sentence) != list:
            sentence = [sentence]
        sentence = self.preprocess(sentence)
        results = self.model(sentence)
        return results

    def preprocess(self,sentence):
        ret = []
        for s in sentence:
            ret.append(self.bpe.segment(s))
        return ret
       
    def proprocess(self,sentence):
        if type(sentence)!=list:
            sentence = [sentence]
        ret = []
        for s in sentence:
            s = self.bpe.decode(s)
            ret.append(s)
        return ret

    def translate(self, sentence):
        if type(sentence) != str:
            raise ValueError('Invalid data type for sentence, must be string!')
        try:
            sentence_lst = self.cut_long_sentence(sentence)
            # before_sent_list = []
            results = self.nmt_model_predict(sentence_lst)
            hypo_text_lst = [[x[1] for x in sub_arr] for sub_arr in results]
            # print(hypo_text_lst)
            prediction_lst = []
            for sub_lst in hypo_text_lst:
                prediction_lst.append(self.proprocess(sub_lst))
            # print(prediction_lst)
            all_prediction = [] 
            for i in range(len(prediction_lst[0])):
                text = ''
                for j in range(len(prediction_lst)):
                    text += prediction_lst[j][i]
                all_prediction.append(text)
            all_prediction = list(set(all_prediction))   
            result = {
                'translation' : all_prediction,
                'code' : '000',
                'msg' : 'success'
            }
        except:
            result = {
                'translation' : None,
                'code' : '001',
                'msg' : 'failure'
            }
        return result

    def release(self):
        # 释放前处理器
        #self.pre_processer.release()
        logging.info('Model released')


def create_noise():
    
    ##################################################################
    ######################### config  ################################
    ##################################################################
    config = {
        'nmt_model_path' : './model/checkpoint_best.pt',
        'dictionary_path' : './data/fairseq/bin',
        'bpe_model_path' : './model/ch_bpe.model',
        'beam_search_width' : 5,
        'nbest' : 5,
    }

    model = Denoiser(config)
    sentence = '本发明涉及一种文本校对错误词库的自动构造方法和装置。该方法包括：构建一个大规模的正确词库表，并将每个词按照在正确词库表中的先后顺序进行编号；针对计算机系统字库中的每一个汉字，构造一系列的字表；创建字字之间的相关度系统矩阵表；依次枚举正确词库表中的每一个词，并针对每一个词中的每个汉字依次进行其他汉字替换，计算替换一个汉字后的错误词语与正确词语的词语匹配相似度；将词语匹配相似度的数值从大到小进行排序，设定词语匹配的相似度阈值，将大于阈值的词语作为候选对象补充至错误词库。本发明能够克服现有技术中错误词表收集过多依赖人工方式、效率低、覆盖面窄以及词库规模受限等缺点，并能够提高文本自动校对的准确率。'
    print(model.translate(sentence))

    
if __name__ == '__main__':
    create_noise()
