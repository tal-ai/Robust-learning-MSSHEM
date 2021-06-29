from train import *
from utils import *
from attack_adv import *
from torch.nn import PairwiseDistance
import os, argparse


def compute_similarity(model, measures, x_lst, device, alpha):
    # the first item in x_lst must be the original text,
    # starting from the second item it can be augmented texts
    _, _, hidden_states = model(x_lst, device)
    embed_layer_hidden = hidden_states[0] # shape=(bs, seq_len, hidden_size) 
    last_layer_hidden = hidden_states[-1] # shape=(bs, seq_len, hidden_size)

    embed_layer_hidden = embed_layer_hidden.sum(dim=1)/embed_layer_hidden.shape[1] # shape=(bs, hidden_size)
    last_layer_hidden = last_layer_hidden.sum(dim=1)/last_layer_hidden.shape[1] # shape=(bs, hidden_size)

    cos_sim_embd_hidden = measures(embed_layer_hidden[0].reshape(1, -1), embed_layer_hidden) # shape=(bs)
    cos_sim_last_hidden = measures(last_layer_hidden[0].reshape(1, -1), last_layer_hidden) # shape=(bs)

    similarity = alpha*cos_sim_embd_hidden + (1-alpha)*cos_sim_last_hidden
    return similarity.cpu().detach().numpy() # value from [-1, 1]

def select_hard_example(model, train_data_lst, device, alpha, num_hard_examples):
    if(num_hard_examples<1):
        assert 'num_hard_examples must be larger than 1'
    hard_example_lst = [[] for _ in range(num_hard_examples)]
    measures = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for item in train_data_lst:
        x_lst = item['text_lst']
        label = item['label']
        sim_info_lst = item['info_lst']

        similarity = compute_similarity(model, measures, x_lst, device, alpha)
        similarity = abs(similarity)
        hard_idx_lst = np.argsort(similarity)[:num_hard_examples] # ascending order, smaller similarity value means larger shift (hard examples)

        for i in range(num_hard_examples):
            try:
                hard_text = x_lst[hard_idx_lst[i]]
                sim_info = sim_info_lst[hard_idx_lst[i]]
            except:
                hard_text = x_lst[0]
                sim_info = 'original'
            example = {'text' : hard_text, 'label' : label, 'info' : sim_info}
            hard_example_lst[i].append(example)
    return hard_example_lst


def helper(data_lst, all_train_data_lst, info):
    for i in range(len(data_lst)):
        item = data_lst[i]
        text = item['text']
        idx = item['idx']
        if(text not in all_train_data_lst[idx]['text_lst']):
            all_train_data_lst[idx]['text_lst'].append(text)
            all_train_data_lst[idx]['info_lst'].append(info)
    return all_train_data_lst


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default='')
    parser.add_argument("--which_data", type=str, default='')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument('--pretrained_model_path', type=str, default='')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--save_hard_example_path', type=str, default='hard_examples/{}')
    parser.add_argument('--best_model_save_path', type=str, default='../model/{}/mining')

    args = parser.parse_args()
    checkpoint = args.checkpoint
    which_data = args.which_data
    device = args.device
    max_iters = args.max_iters
    pretrained_model_path = args.pretrained_model_path
    num_classes = args.num_classes
    save_hard_example_path = args.save_hard_example_path
    best_model_save_path = args.best_model_save_path

    best_model_save_path = best_model_save_path.format(which_data)
    save_hard_example_path = save_hard_example_path.format(which_data)

    
    print('loading all data...')
    train_data_org = load_json('../data/{}/train.json'.format(which_data))['data']
    all_train_data_lst = load_json('../data/{}/train_large.json'.format(which_data))
    valid_data = load_json('../data/{}/valid.json'.format(which_data))['data']


    learning_rate_lst = [5e-7]
    #learning_rate_lst = [5e-8]
    batch_size_lst = [5]
    alpha_lst = [0.5]
    num_hard_examples_lst = [1, 2, 3]
    original_loss_tradeoff_lst = [0.75, 0.50, 1.0] # 1.0 means no stability loss

    standard_loss_fn = F.cross_entropy
    stability_loss_fn = PairwiseDistance(p=2) # L2 distance for stability loss
    
    for learning_rate in learning_rate_lst:
        for batch_size in batch_size_lst:
            for alpha in alpha_lst:
                for num_hard_examples in num_hard_examples_lst:
                    for original_loss_tradeoff in original_loss_tradeoff_lst:
                        is_this_model_trained = False
                        model_prefix = '{}_finetune_lr{}_bs{}_alpha{}_nhard{}_tradeoff{}'.format(checkpoint.split('/')[-1].replace('.pt', ''), \
                        learning_rate, batch_size, alpha, num_hard_examples, original_loss_tradeoff)

                        done_model_lst = [x for x in os.listdir(best_model_save_path)]
                        for done_model in done_model_lst:
                            if(model_prefix in done_model_lst):
                                is_this_model_trained = True
                        if(is_this_model_trained):
                            print('{} is already trained, continue...'.format(model_prefix))
                            continue

                        tmp_save_hard_example_path = os.path.join(save_hard_example_path, model_prefix)
                        print('start finetuning model {}, with mining mode, lr {}, batch_size {}, alpha {}, nhard {}'.format(\
                                                                    model_prefix, learning_rate, batch_size, alpha, num_hard_examples))
                        model = load_model(checkpoint, device)
                        train_helper_mining(model, learning_rate, standard_loss_fn, stability_loss_fn, \
                                        train_data_org, all_train_data_lst, valid_data, batch_size, \
                                        device, max_iters, best_model_save_path, model_prefix, \
                                        save_every_epoch=True, alpha=alpha, num_hard_examples=num_hard_examples, \
                                        original_loss_tradeoff=original_loss_tradeoff, \
                                        save_hard_example_path=tmp_save_hard_example_path)