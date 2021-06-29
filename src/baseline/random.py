import os
import bisect
import json
import numpy as np
from random import shuffle
import math
import random
from Levenshtein import *
import operator


def load_json(path):
    with open(path, 'r') as f:
        js = json.load(f)
    return js

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def read_text(path):
    with open(path, 'r') as f:
        text = f.readlines()
    text = [x.replace('\n','') for x in text]
    text = [x for x in text if len(x.strip())!=0]
    return text

def save_text(text_list, save_path):
    with open(save_path, 'w') as f:
        text_list = [x+'\n' for x in text_list]
        f.writelines(text_list)


# create confusion matrix from parallel data
def count_words(text_list1, text_list2):
    word_counts = {}
    for text in text_list1:
        words = list(str(text))
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
    for text in text_list2:
        words = list(str(text))
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
    word_counts = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True)
    return word_counts

def count_edits(text_list1, text_list2):
    edit_counts = {}
    for text1, text2 in zip(text_list1, text_list2):
        text1 = str(text1)
        text2 = str(text2)
        one_edits = []
        for x in opcodes(text1, text2):
            if x[0] in ['equal', 'replace']:
                sub_text1, sub_text2 = text1[x[1]:x[2]], text2[x[3]:x[4]]
                for word1, word2 in zip(list(sub_text1), list(sub_text2)):
                    edit = word1 + '->' + word2
                    one_edits.append(edit)
                    one_edits.append('null->null')
            elif x[0] == 'insert':
                word1 = 'null'
                sub_text2 = text2[x[3]:x[4]]
                for word2 in list(sub_text2):
                    edit = word1 + '->' + word2
                    one_edits.append(edit)
            elif x[0] == 'delete':
                word2 = 'null'
                sub_text1 = text1[x[3]:x[4]]
                for word1 in list(sub_text1):
                    edit = word1 + '->' + word2
                    one_edits.append(edit)
        for edit in one_edits:
            if edit in edit_counts:
                edit_counts[edit] += 1
            else:
                edit_counts[edit] = 1
    return edit_counts

def create_confusion_matrix(text_list1, text_list2, save_path):
    word_counts = count_words(text_list1, text_list2)
    vocab = [x[0] for x in word_counts] + ['null','WS']
    vocab = [x for x in vocab if x!=' ']
    edit_counts = count_edits(text_list1, text_list2)
    print('共{}种操作'.format(len(edit_counts)))
    matrix_row_list = [' '.join(vocab+[''])]
    for word1 in vocab:
        row_counts = []
        for word2 in vocab:
            edit = word1 + '->' + word2
            if edit in edit_counts:
                row_counts.append(str(edit_counts[edit]))
            else:
                row_counts.append(str(0))
        row_counts.append('')
        matrix_row_list.append(' '.join(row_counts))
    save_text(confusion_matrix, save_path)
    return matrix_row_list


def _normalize_cmx(cmx):
    """
    Normalizes the rows of the confusion matrix, so that they form
    valid probability distributions.
    Assigns zero probability if all elements in a row are zero.
    """
    cmx_row_sums = cmx.sum(axis=1)[:, np.newaxis]
    cmx = np.divide(cmx, cmx_row_sums, out=np.zeros_like(cmx), where=cmx_row_sums!=0)
    return cmx

def _remove_char_from_cmx(c, cmx, lut, vocab):
    """
    Removes a given character from the confusion matrix
    """
    idx = lut.get(c, -1)    
    if idx >= 0:
        cmx = np.delete(cmx, (idx), axis=0) # delete row
        cmx = np.delete(cmx, (idx), axis=1) # delete column
        vocab.pop(idx)
        # LUT must be re-calculated
        lut = make_lut_from_vocab(vocab)
    return cmx, lut, vocab


def load_confusion_matrix(source_path, separator=' '):
    """
    Loads a confusion matrix from a given file. 
    null - token that represents the epsilon character used to define.
    the deletion and insertion operations.
    WS - white-space token.
    Rows represent original tokens, column - perturbed tokens.
    File format:
        - 1st row: vocabulary, e.g., VOCAB a b c ... x y z
        - next |V| rows: rows of the confusion matrix, where V is vocabulary of tokens
    """
    # read input file (e.g., confusion_matrix.txt)
    vocab = None
    cmx = None # confusion matrix    
    matrix = read_text(source_path)
    matrix_row_list = [x.split(separator) for x in matrix]
    for i in range(len(matrix_row_list)):     
        row = matrix_row_list[i]
        if len(row) == 0 or row[0].startswith('#'):
            continue            
        if vocab is None:    
#             print(row)
            row = [c for c in row if len(c) > 0]
            print(row)
            vocab = row
        else:
            row = [c for c in row if len(c) > 0]
            cmx = np.array(row) if cmx is None else np.vstack([cmx, row])
    print(len(vocab))
    cmx = cmx.astype(np.float)
    lut = make_lut_from_vocab(vocab)
    # remove rows and columns of some characters (e.g., WS)
    to_delete = [c for c in ['WS'] if c in lut]
    for c in to_delete:
        cmx, lut, vocab = _remove_char_from_cmx(c, cmx, lut, vocab)        
    cmx = _normalize_cmx(cmx)
    return cmx, lut, vocab

# source_path = 'lib/confusion_matrix/confusion_matrix_ocr.txt'
# cmx, lut, vocab_c  = load_confusion_matrix(source_path)
# np.save('lib/confusion_matrix/cmx/cmx_ocr.npy', cmx)
# save_json(lut, 'lib/confusion_matrix/lut.json')


def make_vocab_from_lut(lut):
    return {v:k for k,v in lut.items()}

def make_lut_from_vocab(vocab):
    return { c:i for i, c in enumerate(vocab) } 


def random_pick(vocab, probabilities): 
    x = random.uniform(0,1) 
#     print('random x: ', x)
    sums = []
    cur_sum = 0
    for proba in probabilities:
        sums.append(cur_sum)
        cur_sum += proba
    idx = bisect.bisect_right(sums, x) - 1
#     print('select idx', idx)
    item = vocab.get(idx)
    return item 

# introduce noise by cmx
def induce_noise_cmx(input_text, cmx, lut):
    """
    Induces noise into the input text using the confusion matrix
    """
    # re-create vocabulary from LUT
    vocab = make_vocab_from_lut(lut)
    n_classes = len(lut)
    input_chars, output_chars = list(input_text), []
    cnt_modifications = 0
    for i in range(len(input_chars) * 2 + 1):
        input_char = input_chars[i // 2] if (i % 2 == 1) else 'null'   
        row_idx = lut.get(input_char, -1)
#         print('input_char, row_idx', input_char, row_idx)
        result_char = input_char
        if row_idx >= 0:
            prob = cmx[row_idx]
            prob_sum = prob.sum()
            if math.isclose(prob_sum, 1.0): 
                result_char = random_pick(vocab, prob)
            else:
                print("Probabilities do not sum to 1 ({}) for row_idx={} (input_char={})!".format(
                    prob_sum, row_idx, input_char
                ))    
        else:
            print("LUT key for '{}' does not exists!".format(input_char))            

        if result_char != 'null':
            output_chars.append(result_char)

        if input_char != result_char:
            # print("{} -> {}".format(input_char, result_char))
            cnt_modifications += 1

    output_text = "".join(output_chars)
    if len(output_text) == 0:
        output_text = input_text
        cnt_modifications = 0
    if input_text != output_text:
        print('input_text={}   output_text={}'.format(input_text, output_text))
    return output_text, cnt_modifications

# introduce noise by confusion matrix
def induce_noise_confusion_matrix(input_text, confusion_matrix, lut, separator=' '):
    """
    Induces noise into the input text using the confusion matrix
    """
    # re-create vocabulary from LUT
    vocab = make_vocab_from_lut(lut)
    matrix_row_list = [x.split(separator) for x in confusion_matrix][1:]
    n_classes = len(lut)
    input_chars, output_chars = list(input_text), []
    cnt_modifications = 0
    for i in range(len(input_chars) * 2 + 1):
        input_char = input_chars[i // 2] if (i % 2 == 1) else 'null'   
        row_idx = lut.get(input_char, -1)
#         print('input_char, row_idx', input_char, row_idx)
        result_char = input_char
        if row_idx >= 0:
            row = matrix_row_list[row_idx]
            prob_list = [float(c) for c in row if len(c) > 0]
            prob_sum = sum(prob_list)
            if math.isclose(prob_sum, 1.0): 
                result_char = random_pick(vocab, prob_list)
            else:
                print("Probabilities do not sum to 1 ({}) for row_idx={} (input_char={})!".format(
                    prob_sum, row_idx, input_char
                ))    
        else:
            print("LUT key for '{}' does not exists!".format(input_char))            

        if result_char != 'null':
            output_chars.append(result_char)

        if input_char != result_char:
            # print("{} -> {}".format(input_char, result_char))
            cnt_modifications += 1

    output_text = "".join(output_chars)
    if len(output_text) == 0:
        output_text = input_text
        cnt_modifications = 0
    if input_text != output_text:
        print('input_text={}   output_text={}'.format(input_text, output_text))
    return output_text, cnt_modifications

# introduce noise for sentence list
def noise_sentences(sent_list, cmx, lut):
    noised_sent_list = []
    for sent in sent_list:
        output_text, cnt_modifications = induce_noise_cmx(sent, cmx, lut)
        noised_sent_list.append(output_text)
    return noised_sent_list

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


# create random confusion matrix
def create_confusion_matrix_random(vocab, save_path, noise_ratio=0.1):
    matrix_row_list = [' '.join(vocab)]
    p = noise_ratio / 3
    print('start create confusion matrix, noise_ratio={}, p={}'.format(
        noise_ratio, p
    ))
    for word1 in vocab:
        row_counts = []
        for word2 in vocab:
            if word1 == word2:       # not change
                if word1 != 'null': 
                    row_counts.append(str(1-2*p))
                else:
                    row_counts.append(str(1-p))
            else:
                if word1 == 'null':     # null->word2  (add)
                    row_counts.append(str(p/(len(vocab)-1)))
                elif word2 == 'null':   # word1->null  (delete)
                    row_counts.append(str(p)) 
                else:     # word1->word2 (replace)
                    row_counts.append(str(p/(len(vocab)-2)))
#         row_counts.append('')
        if len(row_counts) != len(vocab):
            raise ValueError('row_counts != vocab count, row_counts={}, vocab count={}'.format(len(row_counts),len(vocab)))
        matrix_row_list.append(' '.join(row_counts))
    save_text(matrix_row_list, save_path)
    print('matrix_row_list created, row counts = {}, saved to {}'.format(
        len(matrix_row_list), save_path
    ))
    return matrix_row_list

# random confusion matrix to cmx
def load_confusion_matrix_random(source_path, cmx_save_path, lut_save_path, separator=' '):
    vocab = None
    cmx = None # confusion matrix    
    matrix = read_text(source_path)
    print('{} confusion matrix loaded!'.format(source_path))
    matrix_row_list = [x.split(separator) for x in matrix]
    print(len(matrix_row_list[0]), len([x for x in matrix_row_list[0] if len(x)>0]), [x for x in matrix_row_list[0] if len(x)<=0])
    for i in range(len(matrix_row_list)):     
        row = matrix_row_list[i]
        if len(row) == 0 or row[0].startswith('#'):
            continue            
        if vocab is None:    
            row = [c for c in row if len(c) > 0]
#             row = [c[1:-1] for c in row if len(c) > 0]
            vocab = row
            lut = make_lut_from_vocab(vocab)
            save_json(lut, lut_save_path)
            print('saved lut to {}'.format(lut_save_path))
            # print(len(row))
        else:
            row = [c for c in row if len(c) > 0]
            # print(len(row))
            cmx = np.array(row) if cmx is None else np.vstack([cmx, row])
        if i != 0 and i % 200 == 0:
            print('Done {}, cmx shape={}'.format(i, cmx.shape))
    cmx = cmx.astype(np.float)
    np.save(cmx_save_path, cmx)
    print('cmx shape = {}, vocab count = {}, cmx saved to {}'.format(cmx.shape,len(vocab),cmx_save_path))
    return cmx, lut, vocab
  


if __name__ == "__main__":
    text = '我就试试这个'
    confusion_matrix = read_text('../../lib/confusion_matrix/random_matrix_5%.txt'.format(ratio))
    lut = load_json('../../lib/confusion_matrix/lut.json')
    noised_text, cnt = induce_noise_confusion_matrix(text, confusion_matrix, lut)
    noised_text = union_symbol(str(noised_text))
    print(noised_text)