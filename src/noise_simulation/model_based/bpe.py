from utils import *
import sentencepiece as spm


def save_text(lst, path):
    with open(path, 'w') as f:
        for line in lst:
            line = line.replace('\n', '')
            f.write(line+'\n')

def open_text(path):
    with open(path, 'r') as f:
        data = f.readlines()
    return data

def train_bpe_model(raw_text_data_path, model_save_path, vocab_size):
    # if(not os.path.exists(model_save_path)):
    #     os.makedirs(model_save_path)
    data = open_text(raw_text_data_path)
    clean_data = [x for x in data if len(x.replace('\n', ''))>0]
    save_text(clean_data, raw_text_data_path)


    if('.bpe' in model_save_path):
        model_save_path = model_save_path.replace('.bpe', '')
    spm.SentencePieceTrainer.Train('--input={} \
                                    --model_prefix={} --vocab_size={} --model_type=bpe \
                                    --character_coverage=0.9996 --bos_id=1 --eos_id=2 --unk_id=3 --pad_id=0 \
                                    --input_sentence_size=200000 --shuffle_input_sentence=true'.format(raw_text_data_path, model_save_path, vocab_size))
    print('done training bpe model')


def reload_bpe_model(model_path):
    model = spm.SentencePieceProcessor()
    model.Load(model_path)
    return model


def bpe_encode(model, text):
    return model.EncodeAsIds(text)


def bpe_decode(model, sequence):
    return model.DecodeIds(sequence)


def train():
    # 调用bpe算法训练bpe模型
    raw_text_data_path = './data/ch_text.txt'
    model_save_path = './model/ch_bpe'
    vocab_size = 60000
    train_bpe_model(raw_text_data_path, model_save_path, vocab_size)

if __name__ == "__main__":
    train()