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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



load_from_hdf5 = True

class SANVQA(nn.Module):
    '''
    We implement SANVQA based on https://github.com/Cyanogenoid/pytorch-vqa.
    A SAN implementation for show, ask, attend and tell
    Currently as one-hop
    TODO: change to two-hops
    '''

    def __init__(
            self,
            n_class, n_vocab, embed_hidden=300, lstm_hidden=1024, encoded_image_size=7, conv_output_size=2048,
            mlp_hidden_size=1024, dropout_rate=0.5, glimpses=2

    ):  # TODO: change back?  encoded_image_size=7 because in the hdf5 image file,
        # we save image as 3,244,244 -> after resnet152, becomes 2048*7*7
        super(SANVQA, self).__init__()

        #self.vectorizer = nn.Embedding(1200,300).double()

        self.ocr_linear = nn.Sequential(
            nn.Linear(300,300),
            nn.ReLU()
        )
        self.question_linear = nn.Sequential(
            nn.Linear(300,300),
            nn.ReLU()
        )
        for param in self.question_linear.parameters():
            param.requires_grad = False
        #self.ocr_linear = nn.Parameter(torch.tensor(-0.5))
        #self.question_linear = nn.Parameter(torch.tensor(-0.5),requires_grad=False)

        

        #self.cos_similarity = CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, image, question, question_len, embeddings, bboxes, emb_lengths):  # this is an image blind example (as in section 4.1)
        
        batch_size = question.size(0)

        embeddings = self.ocr_linear(embeddings)
        question = self.question_linear(question)

        embeddings = F.normalize(embeddings,p=2,dim=2)
        question = F.normalize(question,p=2,dim=2)

        
        #embeddings = self.ocr_linear * embeddings
        #question = self.question_linear * question

        wordgrid = torch.zeros((batch_size,300,224,224),device=device)
        for batch_i in range(batch_size):
            for emb_i in range(emb_lengths[batch_i]):
                x,y,x2,y2 = bboxes[batch_i,emb_i,:]
                emb_box = embeddings[batch_i,emb_i].repeat((x2-x,y2-y,1)).transpose(2,0)
                #emb_box = emb_box / ((x2-x)*(y2-y))
                wordgrid[batch_i,:,y:y2,x:x2] = emb_box

        #wordgrid = F.normalize(wordgrid,p=2,dim=1)
        wordgrid = wordgrid.view(batch_size,300,-1)

        attention_weights = F.softmax(
            torch.bmm(question,wordgrid),dim=2)

        weighted_avg_wordgrid = torch.sum(wordgrid * attention_weights,dim=2)
        weighted_avg_wordgrid = F.normalize(weighted_avg_wordgrid,p=2,dim=0)
        #output = self.cos_similarity(question.squeeze(),weighted_avg_wordgrid)

        return weighted_avg_wordgrid,question,wordgrid