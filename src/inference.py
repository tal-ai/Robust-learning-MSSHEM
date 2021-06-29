from utils import load_model, load_json
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import argparse, os

def metrics(y_truth_lst, y_hat_lst, y_proba_lst=None):
    assert len(y_truth_lst) == len(y_hat_lst)
    acc = round(accuracy_score(y_truth_lst, y_hat_lst), 4)
    prec = round(precision_score(y_truth_lst, y_hat_lst), 4)
    rec = round(recall_score(y_truth_lst, y_hat_lst), 4)
    f1 = round(2*prec*rec / (prec+rec+1e-10), 4)
    f_05 = round((1+0.5*0.5) *(prec*rec) / (0.5*0.5*prec+rec+1e-10), 4)
    if(y_proba_lst):
        auc = round(roc_auc_score(y_truth_lst, y_proba_lst), 4)
        return acc, prec, rec, f1, auc, f_05
    else:
        return acc, prec, rec, f1, f_05

def inference(checkpoint, test_data_lst, device, bs=100):
    proba = np.zeros(shape=(len(test_data_lst), 2))

    model = load_model(checkpoint, device)
    st = 0
    while(st<len(test_data_lst)):
        ed = min(st+bs, len(test_data_lst))
        batch = test_data_lst[st:ed]
        batch_proba = model(batch, device)[0].cpu().detach().numpy()
        proba[st:ed] = batch_proba
        st = ed
    prediction = (proba[:,1]>0.5).astype(int)
    return prediction, proba[:,1]



if __name__=='__main__':
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default='')
    parser.add_argument("--test_data", type=str, default='')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--batch_size", type=int, default=100)


    args = parser.parse_args()
    checkpoint = args.checkpoint
    test_data = args.test_data
    batch_size = args.batch_size
    device = args.device

    if(os.path.isfile(os.path.join(checkpoint, 'metrics.csv'))):
        done_df = pd.read_csv(os.path.join(checkpoint, 'metrics.csv'))
        done_model_lst = list(done_df['model'])
    else:
        done_df = pd.DataFrame()
        done_model_lst = []

    model_lst, prec_lst, rec_lst, f1_lst, auc_lst, test_lst = [], [], [], [], [], []

    if(not os.path.isfile(checkpoint)):
        root_path = checkpoint
        checkpoint_lst = os.listdir(root_path)
        for c in checkpoint_lst:
            try:
                this_checkpoint = os.path.join(root_path, c)
                if(this_checkpoint in done_model_lst or '.pt' not in c):
                    print('pass {}'.format(this_checkpoint))
                    continue
                for f in ['test.json', 'test_hw_ocr.json', 'test_tal_ocr.json']:
                    filepath = os.path.join(test_data, f)
                    test_json = load_json(filepath)
                    test_data_lst = [x['text'] for x in test_json]
                    y_truth_lst = [int(x['label']) for x in test_json]
                    y_hat, y_proba = inference(this_checkpoint, test_data_lst, device, bs=30)

                    y_hat_lst = [int(x) for x in y_hat]
                    y_proba_lst = [float(x) for x in y_proba]
                    acc, prec, rec, f1, auc, f_05 = metrics(y_truth_lst, y_hat_lst, y_proba_lst)
                    print(this_checkpoint)
                    print(f)
                    print('acc {}, prec {}, rec {}, f1 {}, auc {}'.format(acc, prec, rec, f1, auc))
                    print('-'*66)
                    model_lst.append(this_checkpoint)
                    prec_lst.append(prec)
                    rec_lst.append(rec)
                    f1_lst.append(f1)
                    auc_lst.append(auc)
                    test_lst.append(f)
            except:
                pass
        df = pd.DataFrame()
        df['model'] = model_lst
        df['test'] = test_lst
        df['prec'] = prec_lst
        df['recall'] = rec_lst
        df['f1'] = f1_lst
        df['auc'] = auc_lst

        if(done_df.shape[0]>0):
            df = pd.concat([df, done_df], axis=0)
        df.to_csv(os.path.join(root_path, 'metrics.csv'), index=False)
    else:
        for f in ['test.json', 'test_hw_ocr.json', 'test_tal_ocr.json']:
            filepath = os.path.join(test_data, f)
            test_json = load_json(filepath)
            test_data_lst = [x['text'] for x in test_json]
            y_truth_lst = [int(x['label']) for x in test_json]
            y_hat, y_proba = inference(checkpoint, test_data_lst, device, bs=30)

            y_hat_lst = [int(x) for x in y_hat]
            y_proba_lst = [float(x) for x in y_proba]
            acc, prec, rec, f1, auc, f_05 = metrics(y_truth_lst, y_hat_lst, y_proba_lst)
            print(f)
            print('acc {}, prec {}, rec {}, f1 {}, auc {}'.format(acc, prec, rec, f1, auc))