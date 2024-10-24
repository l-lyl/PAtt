# -*- coding: utf-8 -*-

import numpy as np
import tqdm
import random
from collections import defaultdict

import torch, gc
import torch.nn as nn
from torch.optim import Adam
import torch.nn.init as init
import math

from utils import recall_at_k, ndcg_k, cc_at_k, get_metric, cal_mrr 
from modules import kl_distance, d2s_gaussiannormal, d2s_1overx, kl_distance_matmul

class DPPTrainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, cate_map, kkernel, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.cate_map = cate_map
        self.kkernel = kkernel

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]), flush=True)
        self.criterion = nn.BCELoss()
    def train(self, epoch):  
        self.iteration(epoch, self.train_dataloader, self.kkernel)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, self.kkernel, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, self.kkernel, full_sort, train=False)

    def iteration(self, epoch, dataloader, kkernel, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix, flush=True)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix), None

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg, cc, mrr = [], [], [], 0
        recall_dict_list = []  ##not use in main trainer.valid
        ndcg_dict_list = []
        for k in [5, 10, 15, 20, 40]:
            recall_result, recall_dict_k = recall_at_k(answers, pred_list, k)
            recall.append(recall_result)
            recall_dict_list.append(recall_dict_k)
            
            ndcg_result, ndcg_dict_k = ndcg_k(answers, pred_list, k)
            ndcg.append(ndcg_result)
            ndcg_dict_list.append(ndcg_dict_k)
            
            cc_result = cc_at_k(answers, pred_list, self.cate_map, k, self.args.cate_num)
            cc.append(cc_result)
        mrr, mrr_dict = cal_mrr(answers, pred_list)
        ##"HIT@1": '{:.8f}'.format(recall[0]), "NDCG@1": '{:.8f}'.format(ndcg[0]), change final recall[5] to recall[4] and remove related [5] in return 
        post_fix = {
            "Epoch": epoch,
            "cc@5": '{:.8f}'.format(cc[0]), "HIT@5": '{:.8f}'.format(recall[0]), "NDCG@5": '{:.8f}'.format(ndcg[0]), "F1@5": '{:.8f}'.format(2*(recall[0]+ndcg[0])/2*cc[0]/((recall[0]+ndcg[0])/2+cc[0])), 
            "cc@10": '{:.8f}'.format(cc[1]), "HIT@10": '{:.8f}'.format(recall[1]), "NDCG@10": '{:.8f}'.format(ndcg[1]), "F1@10": '{:.8f}'.format(2*(recall[1]+ndcg[1])/2*cc[1]/((recall[1]+ndcg[1])/2+cc[1])), 
            "cc@15": '{:.8f}'.format(cc[2]), "HIT@15": '{:.8f}'.format(recall[2]), "NDCG@15": '{:.8f}'.format(ndcg[2]), "F1@15": '{:.8f}'.format(2*(recall[2]+ndcg[2])/2*cc[2]/((recall[1]+ndcg[1])/2+cc[2])), 
            "cc@20": '{:.8f}'.format(cc[3]), "HIT@20": '{:.8f}'.format(recall[3]), "NDCG@20": '{:.8f}'.format(ndcg[3]), "F1@20": '{:.8f}'.format(2*(recall[3]+ndcg[3])/2*cc[3]/((recall[1]+ndcg[1])/2+cc[3])), 
            "cc@40": '{:.8f}'.format(cc[4]), "HIT@40": '{:.8f}'.format(recall[4]), "NDCG@40": '{:.8f}'.format(ndcg[4]), "F1@40": '{:.8f}'.format(2*(recall[4]+ndcg[4])/2*cc[4]/((recall[1]+ndcg[1])/2+cc[4])), 
            "MRR": '{:.8f}'.format(mrr)
        }
        print(post_fix, flush=True)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], 2*(recall[3]+ndcg[3])/2*cc[3]/((recall[3]+ndcg[3])/2+cc[3]), recall[2], ndcg[2], recall[3], ndcg[3], recall[4], ndcg[4], mrr], str(post_fix), [recall_dict_list, ndcg_dict_list, mrr_dict]

    def get_pos_items_ranks(self, batch_pred_lists, answers):
        num_users = len(batch_pred_lists)
        batch_pos_ranks = defaultdict(list)
        for i in range(num_users):
            pred_list = batch_pred_lists[i]
            true_set = set(answers[i])
            for ind, pred_item in enumerate(pred_list):
                if pred_item in true_set:
                    batch_pos_ranks[pred_item].append(ind+1)
        return batch_pos_ranks

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name, map_location='cuda:0'))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len, hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len, hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        auc = torch.sum(
            ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc
    
    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class DPPSAModelTrainer(DPPTrainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, cate_map, kkernel, args):
        super(DPPSAModelTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, cate_map, kkernel, args
        )

    def iteration(self, epoch, dataloader, kkernel, full_sort=False, train=True):
        
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        #rec_data_iter = tqdm.tqdm(enumerate(dataloader),
        #                          desc="Recommendation EP_%s:%d" % (str_code, epoch),
        #                          total=len(dataloader),
        #                          bar_format="{l_bar}{r_bar}")
        rec_data_iter = dataloader
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            rec_avg_auc = 0.0
            #for i, batch in rec_data_iter:
            for batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, target_neg, _ = batch
                # Binary cross_entropy
                sequence_output, user_output, _ = self.model.finetune(input_ids, user_ids, kkernel)
                ##user_output2 = user_output.repeat(1,4,1)
                ##sequence_output = (sequence_output+user_output2)/2
                loss, batch_auc = self.cross_entropy(sequence_output, target_pos, target_neg)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                rec_avg_auc += batch_auc.item()
            total_norm = 0
            parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            with open(self.args.gradients_file, 'a') as f:
                f.write( str(epoch) + ': ' + str(total_norm) +" "+ '\n')

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
                "rec_avg_auc": '{:.4f}'.format(rec_avg_auc / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix), flush=True)

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()
            pred_list = None

            if full_sort:
                answer_list = None
                #for i, batch in rec_data_iter:
                i = 0
                for batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output, user_output, _ = self.model.finetune(input_ids, user_ids, kkernel)
                    
                    ##user_output2 = user_output.repeat(1,4,1)
                    ##recommend_output = (recommend_output+user_output2)/2
                    recommend_output = recommend_output[:, -1, :]
                    
                    rating_pred = self.predict_full(recommend_output)
                    
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    ind = np.argpartition(rating_pred, -40)[:, -40:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                    i += 1
                    
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                #for i, batch in rec_data_iter:
                i = 0
                for batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                    i += 1

                return self.get_sample_scores(epoch, pred_list)
