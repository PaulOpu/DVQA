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

from dataset import DVQA, collate_data, transform
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

def init_parameters(mod):
    if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
        #Chargrid: , nonlinearity='relu'
        nn.init.kaiming_uniform_(mod.weight, nonlinearity='relu')
        if mod.bias is not None:
            nn.init.constant(mod.bias, 0)

class Conv2dBatchAct(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act_f, stride=1, padding=0, dilation=1):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Conv2dBatchAct, self).__init__()
        self.conv2d_batch_act = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm2d(out_channels),
            act_f,
        )

        init_parameters(self.conv2d_batch_act[0])


    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        result = self.conv2d_batch_act(x)
        return result

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

        self.vectorizer = nn.Embedding(1200,300).double()
        
        act_f = nn.ReLU()
        #in_channels, out_channels, kernel_size, act_f, stride=1, padding=0, dilation=1
        self.conv1 = Conv2dBatchAct(300,300,1,act_f,1,0,1)

    def forward(self, image, question, question_len, embeddings, bboxes, emb_lengths):  # this is an image blind example (as in section 4.1)
        
        batch_size = question.size(0)
        embeddings = self.vectorizer(embeddings)
        question = self.vectorizer(question)

        wordgrid = torch.zeros((batch_size,300,224,224),device=device,dtype=torch.float64)
        for batch_i in range(batch_size):
            for emb_i in range(emb_lengths[batch_i]):
                x,y,x2,y2 = bboxes[batch_i,emb_i,:]
                
                emb_box = embeddings[batch_i,emb_i].repeat((x2-x,y2-y,1)).transpose(2,0)
                wordgrid[batch_i,:,y:y2,x:x2] = emb_box

        wordgrid = F.normalize(wordgrid,p=2,dim=1)
        question = F.normalize(question,p=2,dim=1)
        #output = (question.view((batch_size,300,1,1)) * wordgrid).max(3)[0]
        #output = output.view(batch_size,-1).max(1)[0] == 1.0

        
        #question = self.qu_linear(question)
        #question = question.view(-1,300,1,1)

        wordgrid = wordgrid.view(batch_size,300,-1)

        #Convolutional
        wordgrid = conv1(wordgrid)

        attention_weights = F.softmax(
            torch.bmm(question,wordgrid),dim=2)

        weighted_avg_wordgrid = torch.sum(wordgrid * attention_weights,dim=2)

        #output = self.cos_similarity(question.squeeze(),weighted_avg_wordgrid)

        return weighted_avg_wordgrid,question