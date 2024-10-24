import torch
import torch.nn as nn
import copy
from modules import Encoder, DPPEncoder, LayerNorm

class DPPSAModel(nn.Module):
    def __init__(self, args):
        super(DPPSAModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.user_embeddings = nn.Embedding(args.num_users, args.hidden_size, padding_idx=0)  ## add for user relevance*diversity*relevance Kernel
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.kernel_position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = DPPEncoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        
        self.device = torch.device('cuda:' + str(self.args.gpu_id))

        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)

    def add_position_embedding(self, sequence, users, batch_kkernel):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        
        batch_emb = batch_kkernel + self.kernel_position_embeddings(position_ids) #new add for ml-1m
        batch_emb = self.LayerNorm(batch_emb)  #need this!
        
        user_embeddings = self.user_embeddings(users)

        return sequence_emb, user_embeddings, batch_emb


    def finetune(self, input_ids, user_ids, kkernel):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        #subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * (-2 ** 32 + 1)
        
        ########### diverse kernel ##################
        b, k = input_ids.shape[0], input_ids.shape[1]
        batch_kkernel = torch.zeros(b, 10, 64)  ##need to change according args
        for n in range(b):
            iid_list = input_ids[n]
            nk_kernel = kkernel[iid_list]
            batch_kkernel[n] = nk_kernel
        
        sequence_emb, user_emb, kernel_emb = self.add_position_embedding(input_ids, user_ids, batch_kkernel.cuda())
        
        ########## call DPPEncoder not DPPAttention; finetune is called by DPPSAModelTrainer iteration ##########
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                user_emb,
                                                extended_attention_mask,
                                                user_ids,
                                                kernel_emb,
                                                output_all_encoded_layers=True)

        sequence_output, user_output, attention_scores = item_encoded_layers[-1]
        return sequence_output, user_output, attention_scores

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
