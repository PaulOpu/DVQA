# -*- coding: utf-8 -*-
import ast
import sys
import pickle
from collections import Counter

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

def attention(q, k, v, mask=None, dropout=None):

    d_k = torch.tensor([k.size(-1)],device=k.device)
    
    scores = torch.bmm(q, k.transpose(-2, -1)) /  torch.sqrt(d_k.float())
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
        
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.bmm(scores, v)
    return output

class CrossSelfAttention(nn.Module):

    def __init__(self,att_heads=1,vector_dim=350,dropout=0.01):
        super(CrossSelfAttention, self).__init__()
        self.vector_dim = vector_dim

        self.wordgrid_key_layer = nn.Conv2d(self.vector_dim,self.vector_dim,3,1,1,bias=False)
        self.wordgrid_query_layer = nn.Conv2d(self.vector_dim,self.vector_dim,3,1,1,bias=False)
        self.wordgrid_value_layer = nn.Conv2d(self.vector_dim,self.vector_dim,3,1,1,bias=False)

        self.question_key_linear_layer = nn.Linear(self.vector_dim,self.vector_dim)
        self.question_query_linear_layer = nn.Linear(self.vector_dim,self.vector_dim)
        self.question_value_linear_layer = nn.Linear(self.vector_dim,self.vector_dim)

        #self.mh_attention_layer = nn.MultiheadAttention(emb_dim+denotion_dim,att_heads,dropout=dropout)
        self.pooling_layer = nn.MaxPool2d(2)

    def forward(self,wordgrid,question):

        #Dimensionality
        batch_size = wordgrid.size(0)
        height = wordgrid.size(2)
        width = wordgrid.size(3)
        n_pixel = height*width
    

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
        wordgrid_key = wordgrid_key.view(batch_size,self.vector_dim,-1).permute(0,2,1)
        wordgrid_query = wordgrid_query.view(batch_size,self.vector_dim,-1).permute(0,2,1)
        wordgrid_value = wordgrid_value.view(batch_size,self.vector_dim,-1).permute(0,2,1)

        #Prepare MultiHead Attention Input
        #(batch size, h*w + q, 350)
        wordgr_qu_key = torch.cat([wordgrid_key,question_key],dim=1)
        wordgr_qu_query = torch.cat([wordgrid_query,question_query],dim=1)
        wordgr_qu_value = torch.cat([wordgrid_value,question_value],dim=1)

        #MultiHead Attention Input
        #output -> (batch size, h*w + q, 350)
        att_output = attention(#self.mh_attention_layer(
            wordgr_qu_key,
            wordgr_qu_query,
            wordgr_qu_value)

        #Separate Wordgrid and Question again
        # (batch size, h*w, 350) and (batch size, q, 350)
        wordgrid,question = torch.split(att_output,n_pixel,dim=1)

        #Reshape and transpose wordgrid
        # (batch size, q, 350) --> (batch size, 350, h, w)
        ##wordgrid = wordgrid.permute(0,2,1).view(
        ##    batch_size,self.vector_dim,height,width,
        ##    )

        ##cont_wordgrid = wordgrid.contiguous()

        #Pooling
        #(batch size, 350, h/2, w/2)
        ##cont_wordgrid = self.pooling_layer(cont_wordgrid)
        
        return question

class SANVQA(nn.Module):
    '''
    We implement SANVQA based on https://github.com/Cyanogenoid/pytorch-vqa.
    A SAN implementation for show, ask, attend and tell
    Currently as one-hop
    TODO: change to two-hops
    '''

    def __init__(self,att_heads=1,denotion_dim=50,emb_dim=300): 
        super(SANVQA, self).__init__()

        self.denotion_dim = denotion_dim
        self.emb_dim = emb_dim
        #0 = wordgrid, 1 = question
        self.denote_word_type_emb = nn.Embedding(2,denotion_dim)

        self.cross_self_attention = CrossSelfAttention(
            att_heads=att_heads,
            vector_dim=emb_dim+denotion_dim,
            dropout=0.01)

    def forward(self, image, question, question_len, embeddings, bboxes, emb_lengths):  # this is an image blind example (as in section 4.1)
        
        device = question.device

        batch_size = question.size(0)
        emb_length_max = embeddings.size(1)
        n_question_word = question.size(1)

        bboxes = bboxes / 4
        h,w = int(224/4),int(224/4)

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
        wordgrid = torch.zeros((batch_size,emb_size,h,w),device=device)
        for batch_i in range(batch_size):
            for emb_i in range(emb_lengths[batch_i]):
                x,y,x2,y2 = bboxes[batch_i,emb_i,:]
                emb_box = embeddings[batch_i,emb_i].repeat((x2-x,y2-y,1)).transpose(2,0)
                #emb_box = emb_box / ((x2-x)*(y2-y))
                wordgrid[batch_i,:,y:y2,x:x2] = emb_box

        #Normalize
        wordgrid = F.normalize(wordgrid,p=2,dim=1)
        question = F.normalize(question,p=2,dim=2)

        #Concat of wordgrid and question to perform self-attention
        self_attended_question = self.cross_self_attention(wordgrid,question)

        ##emb_size = wordgrid.size(1)
        ##h,w = wordgrid.size(2),wordgrid.size(3)
        ##wordgrid = wordgrid.view(batch_size,emb_size,-1)

        self_attended_question = F.normalize(self_attended_question,p=2,dim=2)
        ##wordgrid = F.normalize(wordgrid,p=2,dim=1)

        #new wordgrid and old question
        ##attention_weights = F.softmax(
        ##    torch.bmm(question[:,0,:].view(batch_size,1,-1),wordgrid),dim=2)

        #weighted sum
        ##weighted_avg_wordgrid = torch.sum(wordgrid * attention_weights,dim=2)
        ##weighted_avg_wordgrid = F.normalize(weighted_avg_wordgrid,p=2,dim=0)

        #output = F.sigmoid(self.mlp(question[:,0,:]))

        return self_attended_question,question,wordgrid.view(batch_size,emb_size,h,w)