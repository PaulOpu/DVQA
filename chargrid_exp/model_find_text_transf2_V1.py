# -*- coding: utf-8 -*-
import ast
import sys
import pickle
from collections import Counter
import copy

import torch
import torchvision
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

#from dataset import DVQA, collate_data, transform
import time
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import textwrap
import seaborn as sns

#sys.path.append('/workspace/st_vqa_entitygrid/solution/')
sys.path.append('/project/paul_op_masterthesis/st_vqa_entitygrid/solution/')
from dvqa import enlarge_batch_tensor
from sklearn.metrics import precision_recall_fscore_support

from torchvision.models import resnet101 as _resnet101
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



load_from_hdf5 = True

def attention(q, k, v, d_k, mask=None, dropout=None):

    dim_k = torch.tensor([d_k],dtype=torch.float,device=k.device)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  torch.sqrt(dim_k)#torch.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
        
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.heads = heads

        self.wordgrid_key_layer = nn.Conv2d(self.d_model,self.d_model,3,1,1,bias=False)
        self.wordgrid_query_layer = nn.Conv2d(self.d_model,self.d_model,3,1,1,bias=False)
        self.wordgrid_value_layer = nn.Conv2d(self.d_model,self.d_model,3,1,1,bias=False)

        self.question_key_linear_layer = nn.Linear(self.d_model,self.d_model)
        self.question_query_linear_layer = nn.Linear(self.d_model,self.d_model)
        self.question_value_linear_layer = nn.Linear(self.d_model,self.d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, wordgrid_qu,h,w,mask=None):
        
        bs = wordgrid_qu.size(0)
        wordgrid,question = torch.split(wordgrid_qu,h*w,dim=1)
        wordgrid = wordgrid.permute(0,2,1).view(bs,self.d_model,h,w)

        #Convo 3x3
        wordgrid_key = self.wordgrid_key_layer(wordgrid)
        wordgrid_query = self.wordgrid_query_layer(wordgrid)
        wordgrid_value = self.wordgrid_value_layer(wordgrid)

        #Question Linear
        question_key = self.question_key_linear_layer(question)
        question_query = self.question_query_linear_layer(question)
        question_value = self.question_value_linear_layer(question)

        #Prepare for Attention
        #(batch size, channels, h, w) --> (batch size, h*w, channels) 
        wordgrid_key = wordgrid_key.view(bs,self.d_model,-1).permute(0,2,1)
        wordgrid_query = wordgrid_query.view(bs,self.d_model,-1).permute(0,2,1)
        wordgrid_value = wordgrid_value.view(bs,self.d_model,-1).permute(0,2,1)

        #Prepare MultiHead Attention Input
        #(batch size, h*w + q, 350)
        wordgr_qu_key = torch.cat([wordgrid_key,question_key],dim=1)
        wordgr_qu_query = torch.cat([wordgrid_query,question_query],dim=1)
        wordgr_qu_value = torch.cat([wordgrid_value,question_value],dim=1)

        # perform linear operation and split into h heads
        k = wordgr_qu_key.view(bs, -1, self.heads, self.d_k)
        q = wordgr_qu_query.view(bs, -1, self.heads, self.d_k)
        v = wordgr_qu_value.view(bs, -1, self.heads, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        #MultiHead Attention Input
        #output -> (batch size, h*w + q, 350)
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output

class EncoderLayer(nn.Module):

    def __init__(self,att_heads=1,d_model=350,dropout=0.01):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model

        #self.norm_question1 = Norm(d_model)
        #self.norm_wordgrid1 = Norm(d_model)

        #self.norm_question2 = Norm(d_model)
        #self.norm_wordgrid2 = Norm(d_model)

        #self.norm_question1 = nn.LayerNorm(d_model)
        #self.norm_wordgrid1 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)

        #self.norm_question2 = nn.LayerNorm(d_model)
        #self.norm_wordgrid2 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.multi_head_attention = MultiHeadAttention(
            heads=att_heads, d_model=d_model, dropout = 0.1)


        self.ff = FeedForward(d_model, d_ff=2048, dropout = 0.1)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        #self.mh_attention_layer = nn.MultiheadAttention(emb_dim+denotion_dim,att_heads,dropout=dropout)
        self.pooling_layer = nn.MaxPool2d(2)

    def forward(self,wordgrid_qu,h,w):

        #Dimensionality
        batch_size = wordgrid_qu.size(0)
        #height = wordgrid.size(2)
        #width = wordgrid.size(3)
        n_pixel = h*w

        #wordgrid = wordgrid.view(batch_size,-1,height*width).permute(0,2,1)

        #wordgrid_normed = self.norm_wordgrid1(wordgrid).permute(0,2,1)
        #question_normed = self.norm_question1(question)
        wordgrid_qu_normed = self.norm1(wordgrid_qu)

        att_output = self.multi_head_attention(
            wordgrid_qu_normed,
            h,w,
            mask=None)
        #Separate Wordgrid and Question again
        # (batch size, h*w, 350) and (batch size, q, 350)
        #att_wordgrid,att_question = torch.split(att_output,n_pixel,dim=1)
        #wordgrid,question = wordgrid + att_wordgrid, question + att_question
        wordgrid_qu = wordgrid_qu + self.dropout_1(att_output)

        #wordgrid_normed = self.norm_wordgrid2(wordgrid)
        #question_normed = self.norm_question2(question)
        wordgrid_qu_normed = self.norm2(wordgrid_qu)


        #ff_input = torch.cat([wordgrid_normed,question_normed],dim=1)
        ff_output = self.ff(wordgrid_qu_normed)
        wordgrid_qu = wordgrid_qu + self.dropout_1(ff_output)
        #ff_wordgrid,ff_question = torch.split(ff_output,n_pixel,dim=1)
        #wordgrid,question = wordgrid + ff_wordgrid, question + ff_question
        wordgrid,question = torch.split(ff_output,n_pixel,dim=1)

        #Reshape and transpose wordgrid
        # (batch size, q, 350) --> (batch size, 350, h, w)
        
        wordgrid = wordgrid.permute(0,2,1).view(
            batch_size,self.d_model,h,w
            )

        cont_wordgrid = wordgrid.contiguous()
        cont_question = question.contiguous()

        #Pooling
        #(batch size, 350, h/2, w/2)
        cont_wordgrid = self.pooling_layer(cont_wordgrid)
        h,w = cont_wordgrid.size(2),cont_wordgrid.size(3)
        cont_wordgrid = cont_wordgrid.view(batch_size,self.d_model,-1).permute(0,2,1)
        cont_wordgrid_qu = torch.cat([cont_wordgrid,cont_question],dim=1)
        
        return cont_wordgrid_qu,h,w


class Encoder(nn.Module):
    '''
    We implement SANVQA based on https://github.com/Cyanogenoid/pytorch-vqa.
    A SAN implementation for show, ask, attend and tell
    Currently as one-hop
    TODO: change to two-hops
    '''

    def __init__(self,n_head=5,denotion_dim=50,emb_dim=300,n_layer=1): 
        super(Encoder, self).__init__()

        self.denotion_dim = denotion_dim
        self.emb_dim = emb_dim
        self.d_model = denotion_dim+emb_dim
        self.n_layer = n_layer

        #0 = wordgrid, 1 = question
        self.denote_word_type_emb = nn.Embedding(2,denotion_dim)

        encoder_layer = EncoderLayer(
            att_heads=n_head,
            d_model=self.d_model,
            dropout=0.01)

        self.layers = get_clones(
            encoder_layer, n_layer)

        #self.norm = Norm(self.d_model)
        self.norm = nn.LayerNorm(self.d_model)

        self.classification = nn.Linear(350,2)

    def forward(self, image, question, question_len, embeddings, bboxes, emb_lengths):  # this is an image blind example (as in section 4.1)
        
        device = question.device

        batch_size = question.size(0)
        emb_length_max = embeddings.size(1)
        n_question_word = question.size(1)

        bboxes = bboxes / 4
        height,width = int(224/4),int(224/4)
        h,w = height,width

        #Denote if word is from wordgrid or question
        #(12,50)
        wordgrid_denot,question_denot = self.denote_word_type_emb(
            torch.tensor([0,1],
            device=device))

        #Expand to size of embeddings and question, so that it can be added
        wordgrid_word_denotation = wordgrid_denot.expand(
            (batch_size,emb_length_max,self.denotion_dim))
        question_word_denotation = question_denot.expand(
            (batch_size,n_question_word,self.denotion_dim))

        #Concat all corresponding embeddings and questions
        #(batch size,emb_length_max,300+50)
        embeddings = torch.cat([
            embeddings,
            wordgrid_word_denotation
        ],dim=2)

        #(batch size,n_question_word,300+50)
        question = torch.cat([
            question,
            question_word_denotation],
            dim=2)

        #300+50
        emb_size = embeddings.size(2) 

        #Build Wordgrid
        #wordgrid = torch.zeros((batch_size,emb_size,h,w),device=device)
        wordgrid = torch.zeros((batch_size,h,w,self.d_model),device=device)
        for batch_i in range(batch_size):
            for emb_i in range(emb_lengths[batch_i]):
                x,y,x2,y2 = bboxes[batch_i,emb_i,:]
                #emb_box = embeddings[batch_i,emb_i].repeat((x2-x,y2-y,1)).transpose(2,0)
                #wordgrid[batch_i,:,y:y2,x:x2] = emb_box
                wordgrid[batch_i,y:y2,x:x2] = embeddings[batch_i,emb_i]

        #Normalize
        #wordgrid = F.normalize(wordgrid,p=2,dim=1)
        #question = F.normalize(question,p=2,dim=2)
        wordgrid = wordgrid.view(batch_size,h*w,self.d_model)
        encoder_input = torch.cat([wordgrid,question],dim=1)

        #Concat of wordgrid and question to perform self-attention
        #x,y = wordgrid.clone(),question.clone()
        for i in range(self.n_layer):
            #x,y = self.layers[i](x,y)
            encoder_input,h,w = self.layers[i](encoder_input,h,w)

        #self_attended_question = self.cross_self_attention(wordgrid,question)

        #self_attended_question = F.normalize(y,p=2,dim=2)

        output_normed = self.norm(encoder_input)
        output_wordgrid,question_output = torch.split(output_normed,h*w,1)

        prediction = F.softmax(self.classification(question_output[:,0,:]),dim=1)
        #wordgrid = wordgrid.permute(0,2,1).view(batch_size,self.d_model,height,width)
        output_wordgrid = output_wordgrid.permute(0,2,1).view(batch_size,self.d_model,h,w)
        return prediction,output_wordgrid#.view(batch_size,emb_size,h,w)



class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x,dim=-1):
        norm = self.alpha * (x - x.mean(dim=dim, keepdim=True)) \
        / (x.std(dim=dim, keepdim=True) + self.eps) + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    # def forward(self, wordgrid,question):
    #     bs = wordgrid.size(0)
    #     h,w = wordgrid.size(2),wordgrid.size(3)

    #     wordgrid = wordgrid.view(bs,d_model,-1)
    #     wordgrid = wordgrid.permute(0,2,1)

    #     n_pixel = wordgrid.size(1)

    #     x = torch.cat([wordgrid,question],dim=1)

    #     norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
    #     / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias

    #     wordgrid_norm,question_norm = torch.split(norm,n_pixel,dim=1)

    #     wordgrid_norm = wordgrid_norm.permute(0,2,1).view(bs,-1,h,w)

    #     return wordgrid_norm,question_norm