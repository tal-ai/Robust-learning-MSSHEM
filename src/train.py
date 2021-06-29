import argparse
import os
import time
import pickle
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from model.bert import BertForTextClassification
from attack_adv import gsgr, create_adv_data
from utils import *
from mining import *

def get_batch_valid(data, batch_size, step):
    N = len(data)
    st_idx = int(batch_size*step)
    ed_idx = min((st_idx + batch_size), N)

    inp = [x['text'] for x in data[st_idx : ed_idx]]
    label = [x['label'] for x in data[st_idx : ed_idx]]

    return inp, label

def get_batch(data, batch_size):
    N = len(data)
    selected_idx = np.random.randint(low=0, high=N, size=batch_size)
    inp = [data[x]['text'] for x in selected_idx]
    label = [data[x]['label'] for x in selected_idx]
    return inp, label


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train_model(model, learning_rate, loss_fn, train_data, batch_size, device):
    total_epoch_loss = 0

    model.to(device)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    train_size = len(train_data)
    num_steps = int(train_size//batch_size)
    model.train()
    for step in range(num_steps):
        optim.zero_grad()
        text_batch, label = get_batch(train_data, batch_size)

        proba, logits, hidden_states = model(text_batch, device)
        label_tensor = torch.tensor(label).to(device)
        loss = loss_fn(logits, label_tensor)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        
        total_epoch_loss += loss.item()
        
    return total_epoch_loss/num_steps


def eval_model(model, loss_fn, valid_data, batch_size, device):
    total_epoch_loss = 0
    model.eval()

    valid_size = len(valid_data)
    num_steps = int(valid_size//batch_size)
    with torch.no_grad():
        for step in range(num_steps):
            text_batch, label = get_batch_valid(valid_data, batch_size, step)
            label_tensor = torch.tensor(label).to(device)
            proba, logits, hidden_states = model(text_batch, device)
            loss = loss_fn(logits, label_tensor)
            total_epoch_loss += loss.item()

    return total_epoch_loss/num_steps
	


def train_helper(model, learning_rate, loss_fn, train_data, valid_data, batch_size, \
                    device, max_iters, best_model_save_path, model_prefix, save_every_epoch):
    best_loss = 1e9
    best_model = model
    early_stop_count = 0
    for epoch in range(max_iters):
        if(early_stop_count>6):
            print('early stop at epoch {}'.format(epoch), flush=True)
            break
        train_loss = train_model(model, learning_rate, loss_fn, train_data, batch_size, device)
        val_loss = eval_model(model, loss_fn, valid_data, batch_size, device)

        print('Epoch {}, train loss {}, val loss {}'.format(epoch+1, round(train_loss, 5), round(val_loss, 5)), flush=True)
        # save best valid loss model
        if(val_loss<best_loss):
            early_stop_count = 0
            best_loss = val_loss
            best_model = model

            os.makedirs(best_model_save_path, exist_ok=True)
            with open(os.path.join(best_model_save_path, '{}_best.pt'.format(model_prefix)), 'wb') as f:
                torch.save(best_model, f)
        else:
            early_stop_count += 1
        if(save_every_epoch):
            with open(os.path.join(best_model_save_path, '{}_{}.pt'.format(model_prefix, epoch)), 'wb') as f:
                torch.save(best_model, f)

    print('Training completed...', flush=True)
    print('-'*66, flush=True)
    return best_model

##############################################################################################
################################## hard example mining + stability loss ######################
##############################################################################################
def get_batch_mining(org_data, hard_data, batch_size):
    if(len(org_data)!=len(hard_data)):
        raise ValueError('Size not equal for orginal and hard example data')
    N = len(org_data)
    selected_idx = np.random.randint(low=0, high=N, size=batch_size)
    inp_org = [org_data[x]['text'] for x in selected_idx]
    inp_hard = [hard_data[x]['text'] for x in selected_idx]
    label = [org_data[x]['label'] for x in selected_idx]
    return inp_org, inp_hard, label



def train_model_mining(model, learning_rate, standard_loss_fn, stability_loss_fn, \
                    original_train_data, hard_example_data, batch_size, device, original_loss_tradeoff):
    total_loss = 0
    total_original_loss = 0
    total_stability_loss = 0

    model.to(device)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    train_size = len(original_train_data)
    num_steps = int(train_size//batch_size)
    model.train()

    # hard_example_data: double list, 
    # each item is a list of paris=(simulation noisy text, label, which_noise_somulation_method)
    nhard = len(hard_example_data)

    for i in range(nhard):
        for step in range(num_steps):
            optim.zero_grad()
            text_batch_org, text_batch_noisy, label = get_batch_mining(original_train_data, hard_example_data[i], batch_size)
            label_tensor = torch.tensor(label).to(device)

            _, logits_org, _ = model(text_batch_org, device)
            _, logits_noisy, _ = model(text_batch_noisy, device)

            loss_stand = standard_loss_fn(logits_org, label_tensor) + standard_loss_fn(logits_noisy, label_tensor)
            loss_stablility = stability_loss_fn(logits_org, logits_noisy)
            loss_stablility = loss_stablility.sum()/loss_stablility.shape[0]
            loss = original_loss_tradeoff*loss_stand + (1-original_loss_tradeoff)*loss_stablility

            loss.backward()
            clip_gradient(model, 1e-1)
            optim.step()

            total_loss += loss.item()
            total_original_loss += loss_stand.item()
            total_stability_loss += loss_stablility.item()
        
    total_loss /= (num_steps*nhard)
    total_original_loss /= (num_steps*nhard)
    total_stability_loss /= (num_steps*nhard)
    return total_loss, total_original_loss, total_stability_loss



def train_helper_mining(model, learning_rate, standard_loss_fn, stability_loss_fn, original_train_data, augmented_data_lst, valid_data, batch_size, \
                        device, max_iters, best_model_save_path, model_prefix, save_every_epoch, alpha, num_hard_examples, \
                        original_loss_tradeoff, save_hard_example_path=None):

    best_loss = 1e9
    best_model = model
    early_stop_count = 0

    if(original_loss_tradeoff==1):
        print('Hard example mining is off')
    else:
        print('Hard example mining is on, nhard={}'.format(num_hard_examples))

    for epoch in range(max_iters):
        if(early_stop_count>=5):
            print('early stop at epoch {}'.format(epoch), flush=True)
            break
        
        if(original_loss_tradeoff==1):
            train_loss = train_model(model, learning_rate, standard_loss_fn, original_train_data, batch_size, device)
            val_loss = eval_model(model, standard_loss_fn, valid_data, batch_size, device)
            print('Epoch {}, train loss {}, val loss {}'.format(epoch+1, round(train_loss, 5), round(val_loss, 5)), flush=True)

        else:
            # hard_example_data is a nested list of hard examples
            # each item is a list of paris=(simulation noisy text, label, which_noise_somulation_method)
            hard_example_data = select_hard_example(model, augmented_data_lst, device, alpha, num_hard_examples)
            if(save_hard_example_path!=None):
                os.makedirs(save_hard_example_path, exist_ok=True)
                save_json(hard_example_data, os.path.join(save_hard_example_path, 'hard_example_epoch_{}.json'.format(epoch)))

            train_loss, train_org_loss, train_stability_loss = train_model_mining(model, learning_rate, standard_loss_fn, stability_loss_fn, original_train_data, \
                                                                            hard_example_data, batch_size, device, original_loss_tradeoff)
            val_loss = eval_model(model, standard_loss_fn, valid_data, batch_size, device)
            print('Epoch {}, train loss {}, val loss {}. Standard loss {}, stability loss {}'.format(epoch+1, round(train_loss, 5), \
                    round(val_loss, 5), round(train_org_loss,5), round(train_stability_loss, 5)), flush=True)

        
        # save best valid loss model
        if(val_loss<best_loss):
            early_stop_count = 0
            best_loss = val_loss
            best_model = model

            os.makedirs(best_model_save_path, exist_ok=True)
            with open(os.path.join(best_model_save_path, '{}_best.pt'.format(model_prefix)), 'wb') as f:
                torch.save(best_model, f)
        else:
            early_stop_count += 1

        # save model for every epoch
        if(save_every_epoch):
            print('Save model to {}'.format(best_model_save_path))
            with open(os.path.join(best_model_save_path, '{}_{}.pt'.format(model_prefix, epoch)), 'wb') as f:
                torch.save(best_model, f)

    print('Training completed...', flush=True)
    print('-'*66, flush=True)
    return best_model


def train_robust_bert(pretrained_model_path, data_path, best_model_save_path, device, num_classes=2, max_iters=20, save_every_epoch=False):
    os.makedirs(best_model_save_path, exist_ok=True)
    train_data = load_json(os.path.join(data_path, 'train.json'))['data']
    valid_data = load_json(os.path.join(data_path, 'valid.json'))['data']
    
    print('train size {}, valid size {}'.format(len(train_data), len(valid_data)), flush=True)

    learning_rate_lst = [5e-6]
    batch_size_lst = [10]
    loss_fn = F.cross_entropy

    for learning_rate in learning_rate_lst:
        for batch_size in batch_size_lst:
            done_lst = [x for x in os.listdir(best_model_save_path) if '.pt' in x]
            model_prefix = 'PretrainedBert_lr_{}_bs_{}'.format(learning_rate, batch_size)

            if(model_prefix+'.pt' in done_lst):
                print('Pass model {}'.format(model_prefix), flush=True)
            else:
                print('training mode {}'.format(model_prefix), flush=True)
                model = BertForTextClassification(pretrained_model_path, num_classes, device)
                best_model = train_helper(model, learning_rate, loss_fn, train_data, valid_data, \
                                        batch_size, device, max_iters, best_model_save_path,\
                                        model_prefix, save_every_epoch)