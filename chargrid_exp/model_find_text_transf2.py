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

    def __init__(self,n_head=12,d_model=768,n_layer=4): 
        super(Encoder, self).__init__()

        #self-trained embeddings
        self.vocab_size = 1200
        self.cls_id = torch.tensor([1201])
        self.sep_id = torch.tensor([1202])
        self.word_embedding = nn.Embedding(self.vocab_size+3,d_model)

        #word embedding dimensions
        self.d_model = d_model
        self.n_layer = n_layer

        #0 = wordgrid, 1 = question
        self.denote_word_type_emb = nn.Embedding(2,d_model)

        encoder_layer = nn.TransformerEncoderLayer(self.d_model, n_head)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        #Classification layer results in two possible values
        self.classification = nn.Linear(self.d_model,2)

    def forward(self, image, question,question_ids, question_len, embeddings, bboxes,embedding_ids, emb_lengths):  # this is an image blind example (as in section 4.1)
        batch_size = question.size(0)
        max_emb_length = embeddings.size(1)
        device = question.device

        #Adding CLS and SEP token to question words 
        cls_id = self.cls_id.to(device).expand((batch_size,1))
        sep_id = self.sep_id.to(device).expand((batch_size,1))
        question_ids = torch.cat([cls_id,question_ids,sep_id],dim=1)
        # (batch size, 3, 768)

        #Call Segmentation Embedding (0 = OCR embedding, 1= question embedding)
        wordgrid_denot,question_denot = self.denote_word_type_emb(
            torch.tensor([0,1],
            device=device))

        #Embed Words and add Segmentation Embedding
        embeddings = self.word_embedding(embedding_ids)
        embeddings = embeddings + wordgrid_denot
        #Set vector of empty OCR embedding tokens to 0.
        
        for i in range(batch_size):
            embeddings[i,emb_lengths[i]:] = 0.
            pad_mask[i,:emb_lengths[i]] = 1

        #Embed Question and add Segmentation Embedding
        question = self.word_embedding(question_ids)
        question = question + question_denot

        #Concat Question and OCR Embeddings
        # (batch size, 3 + 12, 768)
        qu_ocr_emb = torch.cat([question,embeddings],dim=1)

        pad_mask = torch.zeros((batch_size,max_emb_length),device=device,dtype=torch.int64)

        qu_ocr_emb = self.encoder_layers(qu_ocr_emb)

        #Linear Layer with CLS token
        prediction = self.classification(qu_ocr_emb[:,0,:])
        # (batch size, 2)
        
        #Check manually if attention is working for the input
        attention_weights = F.softmax(
            torch.bmm(embeddings,question[:,1:2,:].transpose(-1,-2)),dim=1)
        att_vector = torch.sum(embeddings[:,1:,:] * attention_weights,dim=1)
    

        return prediction,att_vector,question[:,1,:]

        # in training file:

            #vqa_model.zero_grad()
            #prediction,output_wordgrid = vqa_model(image, question,question_ids, question_len, embeddings, bboxes,embedding_ids, emb_lengths)

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

