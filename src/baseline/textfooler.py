import os
import sys
import json
import numpy as np
from random import shuffle
import math
import spacy
import jieba
import random
import tensorflow_hub as hub
import tensorflow_text
from scipy.spatial import distance
# sys.path.append('../../')
from model_class.bert import BertForTextClassification
from model_utils import load_model, load_txt, load_json, save_json


# chinese textfooler for classification (2 class) 
class Textfooler(object):
    def __init__(self, model_path, device, pretrained_model_path, model_name, sent_embd_path, word_embd_path='', stopword_path='../../lib/hit_stopwords.txt', num_classes=2):
        # super().__init__()
        self.device = device
        self.model = load_model(model_path, device, pretrained_model_path, model_name, num_classes)
        print('Load classification model {}'.format(model_path))
        self.stopword_lst = load_txt(stopword_path)
        self.sent_embd = hub.load(sent_embd_path)
        # spacy.prefer_gpu()
        if word_embd_path:
            self.nlp = spacy.load(word_embd_path)
        else:
            self.nlp = spacy.load('zh_core_web_lg')
        print('Textfooler init success!')

    def is_model_correct(self, sentence, label, label_proba_dict=None):
        if label_proba_dict is None:
            sentence_lst = [sentence]
            label_proba_dict = {}
            proba = self.model(sentence_lst, self.device)[0].cpu().detach().numpy()
            prediction = (proba[:,1]>0.5).astype(int)
            if prediction[0] == label:
                label_proba_dict = {'label':proba[0][label], 'other':proba[0][1-label]}
                return True, label, label_proba_dict
            return False, label, label_proba_dict
        else:
            return True, label, label_proba_dict

    def get_sentence_vector(self, sentence):
        return np.array(self.sent_embd(sentence)).flatten()

    def spacy_token_to_dict(self, spacy_tokens):
        tokens = []
        for t in spacy_tokens:
            tokens.append({
                'text': t.text,
                'vector': t.vector,
                'pos_': t.pos_
            })
        return tokens

    def get_one_word_importance(self, label, y_pred, label_proba_dict, y_proba_dict):
        importance = 0
        if y_pred == label:
            importance = label_proba_dict['label'] - y_proba_dict['label']
        else:
            importance = (label_proba_dict['label']-y_proba_dict['label']) + (y_proba_dict['other']-label_proba_dict['other'])
        return importance

    def order_word_importance(self, sentence, label, label_proba_dict, bs=10):
        # create dict，key: idx，value: word, sentence, importance
        word_importance = {}
        sentence_lst = []
        # use spacy for segmentation
        spacy_tokens = self.nlp(sentence)
        tokens = self.spacy_token_to_dict(spacy_tokens)
        for i in range(len(tokens)):
            token = tokens[i]
            if i == len(tokens)-1:
                sent_p = ''.join([x['text'] for x in tokens[:i]])
            else:
                sent_p = ''.join([x['text'] for x in tokens[:i]]+[x['text'] for x in tokens[i+1:]])
            sentence_lst.append(sent_p)
            word_importance[str(i)] = {
                'word': token['text'],
                'sentence': sent_p,
                'importance': 0
            }
        # sentence_lst.append(sentence)    
        # predict all sentences
        st = 0
        proba = np.zeros(shape=(len(sentence_lst), 2))
        while(st<len(sentence_lst)):
            ed = min(st+bs, len(sentence_lst))
            batch_text = sentence_lst[st:ed]
            batch_proba = self.model(batch_text, self.device)[0].cpu().detach().numpy()
            proba[st:ed] = batch_proba
            st = ed
        prediction = (proba[:,1]>0.5).astype(int)
        # get importance score
        # label = prediction[-1]
        # label_proba_dict = {'label':proba[-1][label], 'other':proba[-1][1-label]}
        for idx in range(len(sentence_lst)):
            y_pred = prediction[idx]
            y_proba_dict = {'label':proba[idx][label], 'other':proba[idx][1-label]}
            word_importance[str(idx)]['importance'] = self.get_one_word_importance(label, y_pred, label_proba_dict, y_proba_dict)
        word_importance = sorted(word_importance.items(), key=lambda x: x[1]["importance"], reverse=True)
        return word_importance, tokens
    
    def remove_stopwords(self, word_importance, max_try=10):
        new_word_importance = []
        for item in word_importance:
            if item[1]['word'] not in self.stopword_lst:
                new_word_importance.append(item)
        if len(new_word_importance) > max_try:
            new_word_importance = new_word_importance[:max_try]
        return new_word_importance
    
    def find_topN_similar_words(self, word_vector, N=51, thres=0.7):
        # word_vector = self.nlp.get_vector(word)
        similar_words = []
        most_similar = self.nlp.vocab.vectors.most_similar(np.array([word_vector]), n=N)
        words = [self.nlp.vocab[x].text for x in most_similar[0][0][1:]]
        scores = [x for x in most_similar[2][0][1:]]
        for i in range(len(words)):
            if scores[i] >= thres:
                similar_words.append(words[i])
        return similar_words

    def select_similar_sentences(self, sentence, cand_words_and_sents, thres=0.8):
        similar_sents = []
        sent_vector = self.get_sentence_vector(sentence)
        for item in cand_words_and_sents:
            cand_sent = item[1]
            cand_word = item[0]
            cand_sent_vector = self.get_sentence_vector(cand_sent)
            cosine_score = 1 - distance.cosine(sent_vector, cand_sent_vector)
            # print(sentence, item[1], cosine_score)
            if cosine_score >= thres:
                similar_sents.append((cand_word, cand_sent, cosine_score))
        return similar_sents

    def select_word_to_replace(self, tokens, word_idx, topN_words, word_thres):
        word_vector = tokens[word_idx]['vector']
        # topN similar words
        similar_words = self.find_topN_similar_words(word_vector, topN_words, word_thres)
        # print('origin word: {}, similar words: {}'.format(tokens[word_idx]['text'],similar_words))
        # the same postag
        cand_words_and_sents = []
        for word_p in similar_words:
            if word_idx == len(tokens) - 1:
                sentence_p = ''.join([x['text'] for x in tokens[:word_idx]])+word_p
            else:
                sentence_p = ''.join([x['text'] for x in tokens[:word_idx]]+[word_p]+[x['text'] for x in tokens[word_idx+1:]])
            spacy_tokens_p = self.nlp(sentence_p)
            if tokens[word_idx]['pos_'] == spacy_tokens_p[word_idx].pos_:
                cand_words_and_sents.append((word_p, sentence_p))
        # cand_sents = list(set(cand_sents))
        return cand_words_and_sents

    def select_sentence_to_replace(self, sentence, cand_words_and_sents, label, sent_thres, verbose):
        similar_sents = self.select_similar_sentences(sentence, cand_words_and_sents, sent_thres)
        if verbose:
            print('#'*20, 'sents similarity > {}'.format(sent_thres), '#'*20)
            for item in similar_sents:
                print(item)
        # print([x[1] for x in similar_sents])
        sent_changed, sent_not_changed = '', ''
        word_changed, word_not_changed = '', ''
        if len(similar_sents) <= 0:
            return sent_changed, sent_not_changed, word_changed, word_not_changed
        # print('origin sentence={}, label={}'.format(sentence, label))
        similar_proba = self.model([x[1] for x in similar_sents], self.device)[0].cpu().detach().numpy()
        similar_pred = (similar_proba[:,1]>0.5).astype(int)
        # print('similar_pred', similar_pred)
        # print('similar_proba', similar_proba)
        max_score = -1
        min_proba = 99999
        # is_changed = False
        for i in range(len(similar_sents)):
            similar_word = similar_sents[i][0]
            similar_sent = similar_sents[i][1]
            similar_score = similar_sents[i][2]
            if verbose:
                print(similar_sents[i], similar_pred[i], similar_proba[i])
            if similar_pred[i] != label:
                if sent_changed == '' or similar_score > max_score:
                    sent_changed = similar_sent
                    max_score = similar_score 
                    word_changed = similar_word
            else:
                cur_proba = similar_proba[i][label]
                if sent_not_changed == '' or cur_proba < min_proba:
                    sent_not_changed = similar_sent
                    min_proba = cur_proba
                    word_not_changed = similar_word
        return sent_changed, sent_not_changed, word_changed, word_not_changed

    def create_attack_sentence(self, sentence, label, label_proba_dict=None, topN_words=51, word_thres=0.7, sent_thres=0.8, verbose=True):
        # If model prediction is correct, then attack
        is_correct, label, label_proba_dict = self.is_model_correct(sentence, label, label_proba_dict)
        if verbose:
            print('origin sentence: {}, label: {}, model pred correct: {}'.format(
                sentence, label, is_correct
            ))
        if not is_correct:
            return sentence
        # Order word importance 
        word_importance, tokens = self.order_word_importance(sentence, label, label_proba_dict, bs=10)
        # Remove stopwords
        word_importance = self.remove_stopwords(word_importance)
        if verbose:
            print('Ordered words by importance')
            for item in word_importance:
                print('word: {}, importance={}, idx={}'.format(item[1]['word'],item[1]['importance'],item[0]))
        # word_importance = [('1', {'importance':1, 'word':'', 'sentence':''})]
        # for every word in candidates
        replace_words = []
        for item in word_importance:
            word_idx = int(item[0])
            cand_words_and_sents = self.select_word_to_replace(tokens, word_idx, topN_words, word_thres)
            if verbose:
                print('#'*20, 'cand words and cand sents')
                for cand_item in cand_words_and_sents:
                    print(cand_item)
            sent_changed, sent_not_changed, word_changed, word_not_changed = self.select_sentence_to_replace(sentence, cand_words_and_sents, label, sent_thres, verbose)
            if sent_changed != '' and word_changed != '':
                replace_words.append((item[1]['word'],word_changed))
                print('Attack success, replace words: {}, return sentence: {}'.format(replace_words,sent_changed))
                return sent_changed
            elif sent_not_changed == '' and word_not_changed == '':
                continue
            else:
                replace_words.append((item[1]['word'],word_not_changed))
                tokens[word_idx]['text'] = word_not_changed
                if verbose:
                    print('origin word is {}, word_not_changed is {}, sent_not_changed is {}'.format(tokens[word_idx]['text'],word_not_changed,sent_not_changed))
        print('Attack fail, replace words: {}, sentence not change: {}'.format(replace_words,sent_not_changed))
        return sent_not_changed if len(sent_not_changed)>0 else sentence


def predict_all(model, data_lst, device, bs=10):
    sentence_lst = [x['text'] for x in data_lst]
    fool_data = []
    st = 0
    proba = np.zeros(shape=(len(sentence_lst), 2))
    while(st<len(sentence_lst)):
        ed = min(st+bs, len(sentence_lst))
        batch_text = sentence_lst[st:ed]
        batch_proba = model(batch_text, device)[0].cpu().detach().numpy()
        proba[st:ed] = batch_proba
        st = ed
    prediction = (proba[:,1]>0.5).astype(int)
    for i in range(len(prediction)):
        label = data_lst[i]['label']
        if prediction[i] == label:
            one = {k:v for k,v in data_lst[i].items()}
            one['idx'] = i
            one['label_proba_dict'] = {'label':proba[i][label], 'other':proba[i][1-label]}
            fool_data.append(one)
    return fool_data


    
if __name__ == "__main__":
    text = '那还是我从小学时候，有一次，我无意间发现阳台前有一个蚂蚁窝，一群群小蚂蚁在来来往往地忙活着。'
    label = 0
    device = 'cuda'
    model_path = '../../model/pretrained_bert/PretrainedBert_1e-05_8_None.pt'
    pretrained_model_path = '../../model/pretrained_model/chinese_wwm_ext_pytorch'
    sent_embd_path = '../../model/pretrained_model/universal_sentence_encoder'
    word_embd_path = '../../model/pretrained_model/zh_core_web_lg-2.3.1/zh_core_web_lg/zh_core_web_lg-2.3.1'
    textfooler = Textfooler(model_path, device, pretrained_model_path, 'bert', sent_embd_path, word_embd_path)
    attack_text = textfooler.create_attack_sentence(text, label, verbose=False)

