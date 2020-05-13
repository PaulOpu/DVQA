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
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList


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

class Encoder(nn.Module):

    def __init__(self,n_head=5,denotion_dim=50,emb_dim=300,n_layer=2): 
        super(Encoder, self).__init__()

        #dimension for the word type (question or wordgrid word)
        self.denotion_dim = denotion_dim
        #word embedding dimensions
        self.emb_dim = emb_dim
        #fincal word embedding 
        self.d_model = denotion_dim+emb_dim
        self.n_layer = n_layer

        #0 = wordgrid, 1 = question
        self.denote_word_type_emb = nn.Embedding(2,denotion_dim)

        encoder_layer = EncoderLayer(
            att_heads=n_head,
            d_model=self.d_model)

        #duplicate the encoder layers
        self.encoder_layers = _get_clones(encoder_layer, n_layer)

        #Classification layer results in two possible values
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
        #(2,50)
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
            wordgrid_word_denotation],
            dim=2)

        #(batch size,n_question_word,300+50)
        question = torch.cat([
            question,
            question_word_denotation],
            dim=2)


        #Build Wordgrid
        wordgrid = torch.zeros((batch_size,self.d_model,h,w),device=device)
        for batch_i in range(batch_size):
            for emb_i in range(emb_lengths[batch_i]):
                x,y,x2,y2 = bboxes[batch_i,emb_i,:]
                emb_box = embeddings[batch_i,emb_i].repeat((x2-x,y2-y,1)).transpose(2,0)
                wordgrid[batch_i,:,y:y2,x:x2] = emb_box

        #Encoder Loop
        encoded_wordgrid,encoded_question = wordgrid,question
        for mod in self.encoder_layers:
            encoded_wordgrid,encoded_question = mod(encoded_wordgrid,encoded_question)

        #Classification
        prediction = F.softmax(self.classification(encoded_question[:,0,:]),dim=1)
    

        return prediction,encoded_wordgrid

        # in training file:

            #vqa_model.zero_grad()
            #prediction,output_wordgrid = vqa_model(image, question, q_len, embeddings, bboxes, emb_lengths)

            #label = answer.clone()
            ## 6.0 id for "yes", 7.0 id for "no"
            #label[label == 6.0] = 1
            #label[label == 7.0] = 0

            #criterion = nn.CrossEntropyLoss()

            #loss = criterion(prediction, label)
            #loss.backward()
            #optimizer.step()



class EncoderLayer(nn.Module):

    def __init__(self,att_heads=1,d_model=350,dropout=0.01):
        super(EncoderLayer, self).__init__()

        #embedding dimensions
        self.d_model = d_model

        #conv2d wordgrid
        self.wordgrid_layer = nn.Conv2d(self.d_model,self.d_model,3,1,1,bias=False)
        #self.wordgrid_key_layer = nn.Conv2d(self.d_model,self.d_model,3,1,1,bias=False)
        #self.wordgrid_query_layer = nn.Conv2d(self.d_model,self.d_model,3,1,1,bias=False)
        #self.wordgrid_value_layer = nn.Conv2d(self.d_model,self.d_model,3,1,1,bias=False)

        #encoder
        self.encoderlayer = nn.TransformerEncoderLayer(d_model=d_model, nhead=att_heads)
        
        #maxpooling
        self.pooling_layer = nn.MaxPool2d(2)

    def forward(self,wordgrid,question):


        #Dimensionality
        batch_size = wordgrid.size(0)
        height = wordgrid.size(2)
        width = wordgrid.size(3)
        n_pixel = height*width

        #Convo 3x3
        wordgrid = self.wordgrid_layer(wordgrid)
        #wordgrid_key = self.wordgrid_key_layer(wordgrid)
        #wordgrid_query = self.wordgrid_query_layer(wordgrid)
        #wordgrid_value = self.wordgrid_value_layer(wordgrid)

        #Prepare for Attention
        #(batch size, channels, h, w) --> (batch size, h*w, channels) 
        wordgrid = wordgrid.view(batch_size,self.d_model,-1).permute(0,2,1)
        #wordgrid_key = wordgrid_key.view(batch_size,self.d_model,-1).permute(0,2,1)
        #wordgrid_query = wordgrid_query.view(batch_size,self.d_model,-1).permute(0,2,1)
        #wordgrid_value = wordgrid_value.view(batch_size,self.d_model,-1).permute(0,2,1)

        #Prepare MultiHead Attention Input
        #(batch size, h*w + q, 350)
        wordgr_qu = torch.cat([wordgrid,question],dim=1)
        #wordgr_qu_key = torch.cat([wordgrid_key,question],dim=1)
        #wordgr_qu_query = torch.cat([wordgrid_query,question],dim=1)
        #wordgr_qu_value = torch.cat([wordgrid_value,question],dim=1)

        #!!!! EncoderLayer only requires one source and not three values (key,query,value)
        # that's why can only apply one conv2d to wordgrid !!!!
        encoded_wordgrid_qu = self.encoderlayer(wordgr_qu)

        wordgrid,question = torch.split(encoded_wordgrid_qu,n_pixel,dim=1)

        #Reshape and transpose wordgrid
        # (batch size, h*w, 350) --> (batch size, 350, h, w)
        wordgrid = wordgrid.permute(0,2,1).view(
            batch_size,self.d_model,height,width
            )

        cont_wordgrid = wordgrid.contiguous()
        cont_question = question.contiguous()

        #Pooling
        #(batch size, 350, h/2, w/2)
        cont_wordgrid = self.pooling_layer(cont_wordgrid)

        #h,w = cont_wordgrid.size(2),cont_wordgrid.size(3)
        #cont_wordgrid = cont_wordgrid.view(batch_size,self.d_model,-1).permute(0,2,1)
        #cont_wordgrid_qu = torch.cat([cont_wordgrid,cont_question],dim=1)
        
        return cont_wordgrid,cont_question


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

