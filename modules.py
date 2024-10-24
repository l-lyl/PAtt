
import numpy as np

import copy
import math
import torch, gc
import torch.nn as nn
import torch.nn.functional as F
import itertools

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

def kl_distance(mean1, cov1, mean2, cov2):
    trace_part = torch.sum(cov1 / cov2, -1)
    mean_cov_part = torch.sum((mean2 - mean1) / cov2 * (mean2 - mean1), -1)
    determinant_part = torch.log(torch.prod(cov2, -1) / torch.prod(cov1, -1))

    return (trace_part + mean_cov_part - mean1.shape[1] + determinant_part) / 2

def kl_distance_matmul(mean1, cov1, mean2, cov2):
    cov1_det = 1 / torch.prod(cov1, -1, keepdim=True)
    cov2_det = torch.prod(cov2, -1, keepdim=True)
    log_det = torch.log(torch.matmul(cov1_det, cov2_det.transpose(-1, -2)))

    trace_sum = torch.matmul(1 / cov2, cov1.transpose(-1, -2))

    mean_cov_part = torch.matmul((mean1 - mean2) ** 2, (1/cov2).transpose(-1, -2))

    return (log_det + mean_cov_part + trace_sum - mean1.shape[-1]) / 2


def d2s_gaussiannormal(distance, gamma):

    return torch.exp(-gamma*distance)

def d2s_1overx(distance):

    return 1/(1+distance)
    


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    

class Embeddings(nn.Module):
    """Construct the embeddings from item, position.
    """
    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class DPPAttention(nn.Module):
    def __init__(self, args):
        super(DPPAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)  #hidden_size=64, num_attention_heads=2, can add bias=False
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        
        ## diag constant, added to det to avoid error
        K = 2 #tuple
        a_diag = torch.eye(K)*1e-5  #change to 1e-5 when triple with category tmux-6, 1e-3 worse？
        a_diag = a_diag.reshape((1, K, K))
        ## prepare fixed matrix 
        sub_diag = a_diag.repeat(args.max_seq_length*args.max_seq_length, 1, 1)  
        self.sub_diag = sub_diag.repeat(args.batch_size, 1, 1, 1)
        self.device = torch.device('cuda:' + str(args.gpu_id))
        
        #self.user_query = nn.Parameter(torch.rand(1, 10, 64))
        #self.key_query = nn.Parameter(torch.rand(1, 10, 64))

    def elementary_symmetric_polynomial(self, evals, K):
        N = len(evals)
        e = torch.zeros(K+1)
        e[0] = 1
        for n in range(1, N + 1):
            enew = torch.zeros(K + 1)
            enew[0] = 1
            enew[1:] = e[1:] + evals[n-1] * e[:-1]
            e = enew
        return e[-1]
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, user_emb, attention_mask, user_ids, kkernel):
        query_layer = self.query(input_tensor)  ##[256, 64] if user emb is used to represent query
        key_layer = self.key(input_tensor)
        ###key_layer = self.key(kkernel.cuda())
        value_layer = input_tensor
        ## consider one head no need the transpose
        #query_layer = self.transpose_for_scores(mixed_query_layer)
        #key_layer = self.transpose_for_scores(mixed_key_layer)
        #value_layer = self.transpose_for_scores(mixed_value_layer)
        
        #####################################################
        #construct DPP kernel-1:  K=VV^t   V is the item embeddings with trainable query and key
        ####################################################
        q_layer = torch.mul(query_layer, query_layer)
        #k_layer = torch.mul(key_layer, key_layer)
        
        b, k = query_layer.shape[0], query_layer.shape[1]
        a_diag = torch.eye(k)*1e-4
        a_diag = a_diag.reshape((1, k, k))
        batch_diag = a_diag.repeat(b, 1, 1)
        '''
        ######################combine similarity kernel and category-aware kernel######################
        #kkernel = kkernel + batch_diag       #need batch_diag to avoid error
        #q_kernel = torch.bmm(qk_layer, qk_layer.transpose(1,2))  #[256, 10, 10]
        #qk_kernel =  torch.bmm(q_kernel, torch.inverse(kkernel.to(self.device)) + batch_diag.to(self.device)) #+ batch_diag.to(self.device)  
        '''
        ###qk_kernel = torch.bmm(qk_layer, qk_layer.transpose(1,2)) 
        qk_kernel = torch.bmm(q_layer, q_layer.transpose(1,2))
        #qk_kernel = torch.bmm(q_layer, k_layer.transpose(1,2))
        #k_kernel = torch.bmm(k_layer, k_layer.transpose(1,2)) + batch_diag.cuda() 
        #qk_kernel =  torch.bmm(q_kernel, torch.inverse(k_kernel))
        '''
        #####################################################
        #construct DPP kernel-2: quality vs. diversity kernel: 
        #  item-embedding-quality*item-embedding-similarity*item-embedding-quality
        #  referring to  gaussian kernel, 
        #  use trainable query weight and key weight to represent parameters (used to represent quality and similarity)
        #  not good; first try before DPP kernel-1
        ####################################################
        ## calculate X^2
        user_tensor = input_tensor/self.user_query.expand(input_tensor.shape[0], -1, -1)
        q_exponent = torch.bmm(user_tensor, user_tensor.transpose(1,2)) 
        diag_q_exponent = torch.diagonal(q_exponent, dim1=-1, dim2=-2)
        exp_diag_q = torch.exp( -1/8. * diag_q_exponent)
        diag_embed_q = torch.diag_embed(exp_diag_q, dim1=1, dim2=2)
        
        ## calculate (X-Y)^2
        key_tensor = input_tensor/self.key_query.expand(input_tensor.shape[0], -1, -1)
        ZZT = torch.bmm(key_tensor, key_tensor.transpose(1,2)) 
        diag_ZZT = torch.diagonal(ZZT, dim1=1, dim2=2).unsqueeze(2)
        Z_norm_sqr = diag_ZZT.expand_as(ZZT)
        k_exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.transpose(1,2) 
        K_kernel = torch.exp(- 1/8. * k_exponent) #256*10*10
        
        qk_kernel = torch.bmm(torch.bmm(diag_embed_q, ZZT), diag_embed_q) 
        '''
        '''
        #####################################################
        # construct DPP kernel-3: user relevance vs. diversity kernel: 
        # not good; try this before DPP kernel-1
        ####################################################
        user_emb = torch.unsqueeze(query_layer, 1)
        user_rel = torch.exp(torch.squeeze(torch.bmm(user_emb, key_layer.transpose(1,2)))) #input_tensor or key_layer？*5?
        diag_user_rel = torch.diag_embed(user_rel, dim1=1, dim2=2)
        
        ZZT = torch.bmm(key_layer, key_layer.transpose(1,2)) 
        diag_ZZT = torch.diagonal(ZZT, dim1=1, dim2=2).unsqueeze(2)
        Z_norm_sqr = diag_ZZT.expand_as(ZZT)
        k_exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.transpose(1,2) 
        K_kernel = torch.exp(- 1/2. * k_exponent) #256*10*10
        
        qk_kernel = torch.bmm(torch.bmm(diag_user_rel, K_kernel), diag_user_rel) 
        '''
        #####################################################
        #construct DPP kernel-3: user-related user-relevence*diverse-kernel*user-relevence, 
        # but there is no user embeddings here, referring to SSE-PT
        ####################################################
        #qk_layer = torch.mul(query_layer, key_layer) #[256, 10, 64]
        #qk_norm = F.normalize(qk_layer, p=2, dim=2)  
        #qk_kernel = torch.bmm(qk_norm, qk_norm.transpose(1,2))  #[256, 10, 10]
        '''
        #####################################################
        #construct DPP kernel-3: trainable query*diverse-kernel*trainable query 
        ####################################################
        query_layer = self.query.expand(input_tensor.shape[0], -1)
        user_emb = torch.unsqueeze(query_layer, 1)/5
        user_rel = torch.exp(torch.squeeze(torch.bmm(user_emb, input_tensor.transpose(1,2)))) #input_tensor or key_layer？*5?
        diag_user_rel = torch.diag_embed(user_rel, dim1=1, dim2=2)
        
        ZZT = torch.bmm(key_layer, key_layer.transpose(1,2)) 
        diag_ZZT = torch.diagonal(ZZT, dim1=1, dim2=2).unsqueeze(2)
        Z_norm_sqr = diag_ZZT.expand_as(ZZT)
        k_exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.transpose(1,2) 
        K_kernel = torch.exp(- 1/2. * k_exponent) #256*10*10
        
        qk_kernel = torch.bmm(torch.bmm(diag_user_rel, K_kernel), diag_user_rel) 
        '''
        ######################################################calculate DPP weights of any two items ####################################################
        batch_len, seq_len = input_tensor.shape[0], input_tensor.shape[1]
        tuple_subkernel_list = []
        for i in range(seq_len): 
            for j in range(seq_len):
                tuple_set_index = [i, j]  ##diagnal det([[i, i],[i,i]]) = 0
                tuple_subkernel_list.append(qk_kernel[:,tuple_set_index][:,:,tuple_set_index])  #100*256*2*2

        tuple_subkernel = torch.stack(tuple_subkernel_list)
        tuple_subkernel = tuple_subkernel.permute(1, 0, 2, 3).contiguous()  ##256*100*2*2   
        ## subset det , need to add diagonal 1e-5  
        if self.sub_diag.shape[0] !=  tuple_subkernel.shape[0]:
            sub_diag = self.sub_diag[:tuple_subkernel.shape[0], :, :]
        else:
            sub_diag = self.sub_diag # sometimes need sub_diag to avoid model saving error 
        tuple_subdet = torch.linalg.det(tuple_subkernel.cpu() + sub_diag).cuda()   ##256*100 
       
        ## subset probability
        tuple_subprob_list = []
        for b in range(batch_len):
            #evals, evecs = torch.eig(qk_kernel[b].cpu(), eigenvectors=True)  # for normalization
            #real_evals = evals[:, 0]
            #denominator = self.elementary_symmetric_polynomial(real_evals, 2) # k of k_dpp 
            denominator = torch.sum(torch.triu(tuple_subdet[b], diagonal=1)) 
            tuple_subprob = tuple_subdet[b]/denominator.cuda() #torch.exp(-tuple_subdet[b])/denominator.cuda() #tuple_subdet[b]/denominator.cuda()#*seq_len  #torch.exp(-tuple_subdet[b]) #torch.exp(-tuple_subdet[b]/denominator.cuda())#tuple_subdet[b]/denominator.cuda() ##
            tuple_subprob_list.append(tuple_subprob)
        tuple_subprob = torch.stack(tuple_subprob_list)
        
        #### permutations to matirx
        tuple_subprob = tuple_subprob.view(batch_len, seq_len, seq_len) 
        
        #tuple_subprob = torch.mul(tuple_subprob, tuple_subprob)
        #print(tuple_subprob)
        #### add diagal of kernel, as det([[i, i],[i, i]]) = 0, but actually, det(i) = i
        one_diag = torch.diagonal(qk_kernel, dim1=1, dim2=2) #256*10
        one_diag_matrix = torch.diag_embed(one_diag)   #256*10*10
        #batch_diag_matrix = torch.diag_embed(torch.ones(tuple_subprob.shape[1]).to(self.device)).expand(tuple_subprob.shape[0], tuple_subprob.shape[1], tuple_subprob.shape[2])   #256*10*10
        tuple_subprob = -(tuple_subprob + one_diag_matrix)  # -(tuple_subprob + batch_diag_matrix)
        ###identity = torch.eye(seq_len)
        ###identity = identity.unsqueeze(0).expand_as(tuple_subprob).cuda()
        ###tuple_subprob = tuple_subprob * (1 - identity) + identity
        ##tuple_subprob = - tuple_subprob
        ######################################################calculate DPP weights of any two items ####################################################
        
        '''
        ######################################################calculate DPP weights of any two items new version ####################################################
        batch_len, seq_len = input_tensor.shape[0], input_tensor.shape[1]
        combinations = list(itertools.combinations(range(seq_len), 2))
        
        tuple_subprob = torch.zeros(batch_len, seq_len, seq_len).to(self.device)
        
        identity = torch.eye(seq_len)
        identity = identity.unsqueeze(0).expand(batch_len, -1, -1).to(self.device)

        tuple_subprob += identity
        
        for comb in combinations:
            index1, index2 = comb
            tuple_subkernel = qk_kernel[:, [index1, index2]][:, :, [index1, index2]]
            if self.sub_diag.shape[0] !=  tuple_subkernel.shape[0]:
                sub_diag = self.sub_diag[:tuple_subkernel.shape[0], :, :]
            #print(sub_diag.shape, tuple_subkernel.shape, self.sub_diag.shape)
            else:
                sub_diag = self.sub_diag # sometimes need sub_diag to avoid model saving error 
            det_values = torch.linalg.det(tuple_subkernel.cpu() + sub_diag).to(self.device)
            tuple_subprob[:, index1, index2] = det_values
            tuple_subprob[:, index2, index1] = det_values
        
        for b in range(batch_len):
            evals, evecs = torch.eig(qk_kernel[b].cpu(), eigenvectors=True)  # for normalization
            real_evals = evals[:, 0]
            denominator = self.elementary_symmetric_polynomial(real_evals, 2) # k of k_dpp 
            tuple_subprob[b] = tuple_subprob[b].clone()/denominator.to(self.device)
        ######################################################calculate DPP weights of any two items new version####################################################
        '''
        #### attention scores
        tuple_subprob = tuple_subprob / math.sqrt(self.attention_head_size)
        attention_scores = tuple_subprob + torch.squeeze(attention_mask) # attention mask 256*1*10*10; add minus if use VV^T
        ##does we need this? masked value is still near zero after softmax, so we don't need this norm
        attention_probs = nn.Softmax(dim=-1)(attention_scores)   
        attention_probs = self.attn_dropout(attention_probs)
        #print(attention_probs)
        #print(attention_scores.shape)
        context_layer = torch.matmul(attention_probs, value_layer) #256*10*64 if no heads
        #context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        #new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        #context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states, user_emb, attention_probs

    
class DPPIntermediate(nn.Module):
    def __init__(self, args):
        super(DPPIntermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        
        self.dense_user1 = nn.Linear(args.hidden_size, args.hidden_size*4)
        self.dense_user2 = nn.Linear(args.hidden_size*4, args.hidden_size)
        self.LayerNorm_user = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout_user = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor, user_emb):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        
        hidden_user = self.dense_user1(user_emb)
        hidden_user = self.intermediate_act_fn(hidden_user)
        
        hidden_user = self.dense_user2(hidden_user)
        hidden_user= self.dropout_user(hidden_user)
        hidden_user = self.LayerNorm_user(hidden_user + user_emb)

        return hidden_states, hidden_user

    
class DPPLayer(nn.Module):
    def __init__(self, args):
        super(DPPLayer, self).__init__()
        ##self.attention = DPPAttention(args)  
        self.attention = DPPAttention(args)
        self.intermediate = DPPIntermediate(args)

    def forward(self, hidden_states, user_states, attention_mask, user_ids, kkernel):
        attention_output, user_emb, attention_scores = self.attention(hidden_states, user_states, attention_mask, user_ids, kkernel)
        intermediate_output, intermediate_user = self.intermediate(attention_output, user_emb)
        return intermediate_output, intermediate_user, attention_scores


class DPPEncoder(nn.Module):
    def __init__(self, args):
        super(DPPEncoder, self).__init__()
        layer = DPPLayer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, user_states, attention_mask, user_ids, kkernel, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:  ##user started parameters are new added
            hidden_states, hidden_user, attention_scores = layer_module(hidden_states, user_states, attention_mask, user_ids, kkernel)
            if output_all_encoded_layers:
                all_encoder_layers.append([hidden_states, hidden_user, attention_scores])
        if not output_all_encoded_layers:
            all_encoder_layers.append([hidden_states, hidden_user, attention_scores])
        return all_encoder_layers

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)  ## Layer called self-attention
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states, attention_scores = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append([hidden_states, attention_scores])
        if not output_all_encoded_layers:
            all_encoder_layers.append([hidden_states, attention_scores])
        return all_encoder_layers
