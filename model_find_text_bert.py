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

from transformers import BertForSequenceClassification, BertConfig


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

    def __init__(self,n_vocab=1200,n_head=1,d_model=768,n_layer=1,ff_neurons=3072,max_pos=512,num_labels=2): 
        super(Encoder, self).__init__()

        #self-trained embeddings
        self.n_vocab = n_vocab
        self.d_model = d_model
        self.n_layer = n_layer

        self.cls_id = torch.tensor([n_vocab+1])
        self.sep_id = torch.tensor([n_vocab+2])
        #self.word_embedding = nn.Embedding(self.vocab_size+3,d_model)

        config = BertConfig(
            n_vocab+3,output_attentions=True,num_hidden_layers=n_layer,
            max_position_embeddings=max_pos,intermediate_size=ff_neurons,
            hidden_size=d_model,num_attention_heads=n_head,num_labels=num_labels)
        
        self.bert = BertForSequenceClassification(config)#BertModel(config)

        #self.classification = nn.Linear(self.d_model,1)

    def forward(self, image, question,question_ids, question_len, embeddings, bboxes,embedding_ids, emb_lengths, labels):  # this is an image blind example (as in section 4.1)

        bert_input,token_type_ids,position_ids,attention_mask = self.prepare_bert_input(
            question_ids,question_len,embedding_ids,emb_lengths)

        

        #last_hidden_state,pooler_output,attentions = self.bert.forward(
        loss,logits,attentions = self.bert.forward(
            input_ids=bert_input,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            labels=labels.float()
        )

        return loss,logits

        #Linear Layer with CLS token
        prediction = self.classification(pooler_output)
        # (batch size, 2)
        
        #Check manually if attention is working for the input
        attention_weights = F.softmax(
            torch.bmm(embeddings,question[:,0:1,:].transpose(-1,-2)),dim=1)
        att_vector = torch.sum(embeddings * attention_weights,dim=1)
    

        return prediction,att_vector,question[:,0,:]

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

    def prepare_bert_input(self,question_ids,question_len,embedding_ids,emb_lengths):
        batch_size = question_ids.size(0)
        max_qu_words = question_ids.size(1)
        max_emb_words = embedding_ids.size(1)
        device = question_ids.device

        #concat like CLS + question + SEP + embeddings + SEP
        cls_token = self.cls_id.to(device).expand((batch_size,1))
        sep_token = self.sep_id.to(device).expand((batch_size,1))

        bert_input = torch.cat([
            cls_token,
            question_ids,
            sep_token,
            embedding_ids,
            sep_token
        ],dim=1)

        #token_type_ids = CLS,question,SEP = 0 and embeddings,SEP = 1
        bert_input_size = bert_input.size(1)
        token_type_ids = torch.zeros((batch_size,bert_input_size),device=device,dtype=torch.int64)
        token_type_ids[:,(max_qu_words+2):] = 1

        #position_ids = range(0,length(question+embeddings+3))
        position_ids = torch.arange(bert_input_size,device=device).expand((batch_size,bert_input_size))

        #attention_mask = index of embeddings < emb_lengths = 0
        qu_att_mask = (torch.arange(max_qu_words,device=device)[None, :] < question_len[:, None]).long()
        emb_att_mask = (torch.arange(max_emb_words,device=device)[None, :] < emb_lengths[:, None]).long()
        special_mask = torch.tensor([1],device=device).expand((batch_size,1))
        attention_mask = torch.cat([
            special_mask,
            qu_att_mask,
            special_mask,
            emb_att_mask,
            special_mask
        ],dim=1)

        return bert_input,token_type_ids,position_ids,attention_mask



