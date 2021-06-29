from train import *
from utils import *
from attack_adv import *
from torch.nn import PairwiseDistance
import os, argparse, random


def get_batch_nat(org_data, noisy_data, batch_size):
    if(len(org_data)!=len(noisy_data)):
        raise ValueError('Size not equal for orginal and noisy example data')
    N = len(org_data)
    selected_idx = np.random.randint(low=0, high=N, size=batch_size)
    inp_org = [org_data[x]['text'] for x in selected_idx]
    inp_noisy = [random.choice(noisy_data[x]['text']) for x in selected_idx]
    label = [org_data[x]['label'] for x in selected_idx]
    return inp_org, inp_noisy, label

def train_model_nat(model, learning_rate, standard_loss_fn, stability_loss_fn, \
                    original_train_data, noisy_train_data, batch_size, device, original_loss_tradeoff):
    total_loss = 0
    total_original_loss = 0
    total_stability_loss = 0

    model.to(device)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    train_size = len(original_train_data)
    num_steps = int(train_size//batch_size)
    model.train()

    for step in range(num_steps):
        optim.zero_grad()
        text_batch_org, text_batch_noisy, label = get_batch_nat(original_train_data, noisy_train_data, batch_size)
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
        
    total_loss /= num_steps
    total_original_loss /= num_steps
    total_stability_loss /= num_steps
    return total_loss, total_original_loss, total_stability_loss



def train_helper_nat(model, learning_rate, standard_loss_fn, stability_loss_fn, original_train_data, noisy_train_data, valid_data, batch_size, \
                        device, max_iters, best_model_save_path, model_prefix, save_every_epoch, original_loss_tradeoff):
    best_loss = 1e9
    best_model = model
    early_stop_count = 0

    for epoch in range(max_iters):
        if(early_stop_count>=5):
            print('early stop at epoch {}'.format(epoch), flush=True)
            break

        train_loss, train_org_loss, train_stability_loss = train_model_nat(model, learning_rate, standard_loss_fn, \
                                                stability_loss_fn, original_train_data, noisy_train_data, batch_size, device, original_loss_tradeoff)

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



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default='')
    parser.add_argument("--which_data", type=str, default='')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument('--pretrained_model_path', type=str, default='')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--best_model_save_path', type=str, default='../model/{}/NAT')

    args = parser.parse_args()
    checkpoint = args.checkpoint
    which_data = args.which_data
    device = args.device
    max_iters = args.max_iters
    pretrained_model_path = args.pretrained_model_path
    num_classes = args.num_classes
    best_model_save_path = args.best_model_save_path


    best_model_save_path = best_model_save_path.format(which_data)
    os.makedirs(best_model_save_path, exist_ok=True)
    
    print('This is baseline NAT, loading original and rule-based noisy data...')
    original_train_data = load_json('../data/{}/train.json'.format(which_data))['data']
    valid_data = load_json('../data/{}/valid.json'.format(which_data))['data']
    noisy_train_data_raw = load_json('path_to_rule_based_samples')

    noisy_train_data = [{'text' : [x['text']], 'label' : x['label']} for x in original_train_data]
    for item in noisy_train_data_raw:
        idx = item['idx']
        noisy_train_data[idx]['text'].append(item['text'])

    learning_rate_lst = [5e-8, 5e-7]
    batch_size_lst = [5]
    original_loss_tradeoff_lst = [0.75, 0.50, 1.0] # 1.0 means no stability loss

    standard_loss_fn = F.cross_entropy
    stability_loss_fn = PairwiseDistance(p=2) # L2 distance for stability loss
    
    for learning_rate in learning_rate_lst:
        for batch_size in batch_size_lst:
            for original_loss_tradeoff in original_loss_tradeoff_lst:
                is_this_model_trained = False
                model_prefix = 'NAT_{}_finetune_lr{}_bs{}_tradeoff{}'.format(checkpoint.split('/')[-1].replace('.pt', ''), \
                learning_rate, batch_size, original_loss_tradeoff)

                done_model_lst = [x for x in os.listdir(best_model_save_path)]
                for done_model in done_model_lst:
                    if(model_prefix in done_model_lst):
                        is_this_model_trained = True
                if(is_this_model_trained):
                    print('{} is already trained, continue...'.format(model_prefix))
                    continue

                print('start finetuning NAT {}, with lr {}, batch_size {}'.format(\
                                                            model_prefix, learning_rate, batch_size))
                model = load_model(checkpoint, device)

                train_helper_nat(model, learning_rate, standard_loss_fn, stability_loss_fn, \
                                original_train_data, noisy_train_data, valid_data, batch_size, \
                                device, max_iters, best_model_save_path, model_prefix, \
                                save_every_epoch=True, original_loss_tradeoff=original_loss_tradeoff)