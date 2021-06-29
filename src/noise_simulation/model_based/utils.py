import numpy as np
import tensorflow as tf
import math
import pickle
from pypinyin import pinyin, lazy_pinyin, Style
#import zmail
import json

def get_position_encoding(
    length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """
    Return positional encoding.
    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.
    Args:
        length: Sequence length.
        hidden_size: Size of the
        min_timescale: Minimum scale that will be applied at each position
        max_timescale: Maximum scale that will be applied at each position
    Returns:
        Tensor with shape [length, hidden_size]
    """
    # We compute the positional encoding in float32 even if the model uses
    # float16, as many of the ops used, like log and exp, are numerically unstable
    # in float16.
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.cast(num_timescales, tf.float32) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal

def zero_padding_mask(x):
    '''
    input:
    x: a sequence of shape [batch_size * seq_len]
    
    returns:
    a tensor of shape [batch_size * 1 * 1 * seq_len]
    where 1 indicates zero padding, 0 indicates otherwise 

    '''
    mask = tf.cast(tf.equal(x, 0), tf.float32) # batch_size * seq_len
    # add extra dimension because mask will be added to attention logits, 
    # whose shape is [... * seq_len_q * seq_len_k]
    return mask[:, tf.newaxis, tf.newaxis, :] # batch_size * 1 * 1 * seq_len

def create_look_ahead_mask(size):
    '''
    The look-ahead mask is used to mask the future tokens in a sequence. In other words, the mask indicates which entries should not be used.
    This means that to predict the third word, only the first and second word will be used. 
    Similarly to predict the fourth word, only the first, second and the third word will be used and so on.
    '''
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0) # 1 means don't use, 0 means use
    return mask  # (seq_len, seq_len)


def get_batch_data(data, batch_size):
    data_size = len(data['src_inp'])
    batch_idx = np.random.randint(low=0, high=data_size, size=(batch_size))
    batch_data = (data['src_inp'][batch_idx], data['tar_inp'][batch_idx], data['tar_out'][batch_idx])
    return batch_data


# load a pickle object
def load_data(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    for k in data.keys():
        data[k] = np.array(data[k], dtype=np.float32)
    return data

def load_pickle(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

    
def save_pickle(data, data_path):
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)

def load_json(path):
    with open(path, 'r') as handle:
        data = json.load(handle)
    return data

def save_json(data, path):
    with open(path, 'w') as handle:
        json.dump(data, handle)

def char_2_pinyin(text):
    pinyin = lazy_pinyin(text)
    return pinyin[0].capitalize()+' '+''.join(pinyin[1:]).capitalize() if len(pinyin)>1 else pinyin[0].capitalize()

