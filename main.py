# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import torch, gc
gc.collect()
torch.cuda.empty_cache()
import argparse
import time

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset
from trainers import DPPSAModelTrainer
from models import S3RecModel
from seqmodels import DPPSAModel
from utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed, get_cates_map, compute_distance_matrix

import pickle as cPickle
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='beauty', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=10, type=int, help="pretrain epochs 10, 20, 30...")

    # model args
    parser.add_argument("--model_name", default='Finetune_full', type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.4, help="hidden dropout p for generative Trans")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=20, type=int)
    parser.add_argument('--distance_metric', default='wasserstein', type=str)
    parser.add_argument('--pvn_weight', default=0.1, type=float)
    parser.add_argument('--kernel_param', default=1.0, type=float)
    parser.add_argument('--diverse_kernel', default=0, type=int, help="0:pre-learned diversity kernel; 1:Levenshtein distance")
    
    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=41, type=int)
    
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    args = parser.parse_args()

    set_seed(args.seed)   
    check_path(args.output_dir)

    os.environ["PYTORCH_NO_CUDA_MEMORY_ALLOCATOR"] = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.data_name + '.txt'
    
    ########################## cc and diverse kernel #######################################
    args.cate_file = args.data_dir + args.data_name + '/cate.txt'  
    cate_map, cate_num, list_of_cates = get_cates_map(args.cate_file) ##
    args.cate_num = cate_num
    if args.diverse_kernel == 0:
        diverse_emb_file = args.data_dir + args.data_name + '/' + 'item_kernel_3.pkl'
        lk_param = cPickle.load(open(diverse_emb_file, 'rb'), encoding="latin1")
        lk_tensor = torch.FloatTensor(lk_param['V']) #|V|*64
    
        average = torch.mean(lk_tensor[:100], dim=0) ##add a new tensor at the start of the kernel, as 0 item 
        new_lk_tensor = torch.cat((average.unsqueeze(0), lk_tensor), dim=0) 
        lk_emb_i = F.normalize(new_lk_tensor, p=2, dim=1) #for anime and beauty
        #k_kernel = torch.matmul(lk_emb_i, lk_emb_i.t())  #|V|+1 * |V|+1
        #k_kernel = new_lk_tensor   
        k_kernel = torch.exp(new_lk_tensor)  #add extra sigmoid 
        #k_kernel = F.normalize(k_kernel, p=2, dim=1)
        #k_kernel = lk_emb_i   
    elif args.diverse_kernel == 1:
        distance_file = args.data_dir + args.data_name + '/' + 'cate_distance.npy'
        k_kernel = compute_distance_matrix(list_of_cates, distance_file)
        
    #############################################################################################
    
    ## item id starts from 1
    user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users = \
        get_user_seqs(args.data_file)
    #item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.num_users = num_users
    args.mask_id = max_item + 1
    
    args_str = f'{args.model_name}-{args.data_name}-{args.hidden_size}-{args.num_hidden_layers}-{args.num_attention_heads}-{args.hidden_act}-{args.attention_probs_dropout_prob}-{args.hidden_dropout_prob}-{args.max_seq_length}-{args.lr}-{args.weight_decay}-{args.ckp}-{args.kernel_param}-{args.pvn_weight}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')
        
    args.weights_file = os.path.join(args.output_dir, args_str + '-weights-mask.txt')
    args.gradients_file = os.path.join(args.output_dir, args_str + '-gradients.txt')

    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)
    
    train_dataset = SASRecDataset(args, user_seq, data_type='train')
    eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
    test_dataset = SASRecDataset(args, user_seq, data_type='test')
        
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
    eval_sampler = SequentialSampler(eval_dataset)
    test_sampler = SequentialSampler(test_dataset)
    #test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=200)
    
    model = DPPSAModel(args=args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=100)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=100)
    trainer = DPPSAModelTrainer(model, train_dataloader, eval_dataloader,
                                test_dataloader, cate_map, k_kernel, args)

    if args.do_eval:   ## call pre-learned model
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info, _ = trainer.test(0, full_sort=True)
        print(result_info)
    else:
        score_list = []              
        early_stopping = EarlyStopping(args.checkpoint_path, patience=50, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch) 
            scores, _, _ = trainer.valid(epoch, full_sort=True)
            score_list.append(scores[4])
            if epoch > 0: #30
                early_stopping([scores[1]], trainer.model) # previous is scores[-1:], DPPAttention change to [scores[3]] for ml-1m
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                
        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model, save in earyly-stopping
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        valid_scores, _, _ = trainer.valid('best', full_sort=True)
        trainer.args.train_matrix = test_rating_matrix
        scores, result_info, _ = trainer.test('best', full_sort=True)
        
    print(args_str)
    #print(score_list)
    
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
main()