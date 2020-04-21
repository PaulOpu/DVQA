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

class Attention(nn.Module):  # SANVQAbeta
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.5):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x))
        return x

def apply_attention(input, attention):  # softmax weight, then weighted average
    """ Apply any number of attention maps over the input. """
    n, c = input.size()[:2]
    glimpses = attention.size(1)  # = 2

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, 1, c, -1)  # [n, 1, c, spatial]
    attention = attention.view(n, glimpses, -1)
    attention = F.softmax(attention, dim=-1).unsqueeze(2)  # [n, glimpses, 1, spatial]
    weighted = attention * input  # [n, glimpses, channel, spatial]
    weighted_mean = weighted.sum(dim=-1)  # [n, glimpses, channel]
    return weighted_mean.view(n, -1)

def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    # input(feature_map.shape)
    # input(feature_vector.shape)
    spatial_size = feature_map.dim() - 2
    # tiled = feature_vector.view(n, c, *([1] * spatial_size)).repeat(1, int(feature_map.size()[1]/c), 1, 1).expand_as(
    #     feature_map)
    tiled = feature_vector.view(n, c, *([1] * spatial_size)).repeat(1, 1, feature_map.size()[-2],
                                                                    feature_map.size()[-1])
    return tiled


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

class Chargrid_Encoder(torch.nn.Module):
    def __init__(self, in_channels, act_f):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Chargrid_Encoder, self).__init__()

        encoder_parameter_blocks= [
            #in, out, stride, dilation
            [in_channels,64,2,1], #a #56
            [64,128,2,1], #a #28
            [128,256,2,2], #a #14
            [256,512,1,4], #a''
            [512,512,1,8], #a''
        ]

        self.encoder_modules = nn.ModuleList()

        for i,(in_ch,out_ch,stride,dilation) in enumerate(encoder_parameter_blocks):
            self.encoder_modules.extend([
                Conv2dBatchAct(in_ch,out_ch,3,act_f,stride,dilation,dilation), #C112
                Conv2dBatchAct(out_ch,out_ch,3,act_f,1,dilation,dilation),
                Conv2dBatchAct(out_ch,out_ch,3,act_f,1,dilation,dilation),
                torch.nn.Dropout()
                ]
            )

        self.lateral_start = [3,7,11]
        

        decoder_parameter_blocks= [
            #in, out, kernel, stride list, dilation
            [256+512,256,"b"],
            [256+128,128,"b"],
            [128+64,64,"c"]
        ]

        self.decoder_modules = nn.ModuleList()

        for i,(in_ch,out_ch,block_type) in enumerate(decoder_parameter_blocks):
            curr_module = [
                Conv2dBatchAct(in_ch,in_ch,1,act_f),
                nn.ConvTranspose2d(in_ch,out_ch,3,2,1,1),
                ]
            
            if block_type == "b":
                curr_module += [
                    Conv2dBatchAct(out_ch,out_ch,3,act_f,1,1),
                    Conv2dBatchAct(out_ch,out_ch,3,act_f,1,1),
                    torch.nn.Dropout()
                ]

            

            self.decoder_modules.extend(curr_module)
            
        self.lateral_end = [0,5,10]

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        encoder_results = []
        for i,encoder_block in enumerate(self.encoder_modules):
            x = encoder_block(x)
            if i in self.lateral_start:
                encoder_results.append(x)

        curr_lateral = 0
        for i,decoder_block in enumerate(self.decoder_modules):
            if i in self.lateral_end:
                x = torch.cat([x,encoder_results[2-curr_lateral]],dim=1)
                curr_lateral += 1
            x = decoder_block(x)
        
        return x

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

        #just concat
        #conv_output_size  = 64 #* 2 #2048
        #lstm_hidden = 16 #512
        self.chargrid_channels = 0
        mid_feature = 512 #128
        #mlp_hidden_size = 128 # 1024
        #embed_hidden = 50
        #glimpses = 2

        self.n_class = n_class
        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, lstm_hidden, batch_first=True)

        self.enc_image_size = encoded_image_size

        
        resnet = torchvision.models.resnet101(
            pretrained=True)  
        modules = list(resnet.children())[:-2] #:-6]
        if self.chargrid_channels > 0:
            modules.append(nn.ConvTranspose2d(2048,2048,3,2,1,1))
        ##modules[0] = nn.Conv2d(3, conv_output_size, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        ##modules[1] = nn.BatchNorm2d(conv_output_size)
        self.resnet = nn.Sequential(*modules)

        act_f = nn.ReLU()
        #Chargrid
        if self.chargrid_channels > 0:
            print("Using Chargrid")

            #chargrid_resnet = torchvision.models.resnet101(
            #    pretrained=False)
            #chargrid_modules = list(chargrid_resnet.children())[:-6] #[:-2]
            #chargrid_modules[0:0] = [
            chargrid_modules = [
                Conv2dBatchAct(41,16,1,act_f,1), 
                Conv2dBatchAct(16,32,3,act_f,2,1), # -> 112
                Conv2dBatchAct(32,64,3,act_f,2,1), # -> 56
                Chargrid_Encoder(64,act_f),
                Conv2dBatchAct(64,128,3,act_f,2,1), # -> 28
                Conv2dBatchAct(128,256,3,act_f,2,1), # -> 14
                ]
            #chargrid_modules.append(Chargrid_Encoder(64,act_f))
            self.chargrid_net = nn.Sequential(*chargrid_modules)
            #self.init_parameters(self.chargrid_net)
            
            entitygrid_depth = 2048+256
            self.entitygrid_net = nn.Sequential(
                Conv2dBatchAct(entitygrid_depth,entitygrid_depth,1,act_f,1),
                Conv2dBatchAct(entitygrid_depth,entitygrid_depth,1,act_f,1),
                Conv2dBatchAct(entitygrid_depth,entitygrid_depth,1,act_f,1),
                Conv2dBatchAct(entitygrid_depth,3072,3,act_f,2,1),
            )

        #chargrid_modules[0][0].conv1 = nn.Conv2d(
        #    1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        
        #entitygrid_modules[0][0].downsample[0] = nn.Conv2d(
        #    1024, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        #self.chargrid_net = nn.Sequential(
        #    nn.Conv2d(64, mid_features, 1)
        #)

        
        self.dropout = nn.Dropout(
            p=dropout_rate)  # insert Dropout layers after convolutional layers that you’re going to fine-tune

        self.attention = Attention(self.chargrid_channels+ conv_output_size, lstm_hidden, mid_features=mid_feature, glimpses=glimpses, drop=0.5)

        self.mlp = nn.Sequential(self.dropout,  ## TODO: shall we eliminate this??
                                 nn.Linear((self.chargrid_channels + conv_output_size) * glimpses + lstm_hidden,
                                           mlp_hidden_size),
                                 nn.ReLU(),
                                 self.dropout,
                                 nn.Linear(mlp_hidden_size, self.n_class))

        #self.apply(self.init_parameters)
        self.fine_tune()  # define which parameter sets are to be fine-tuned

        self.hop = 1

    def forward(self, image, question, question_len, chargrid):  # this is an image blind example (as in section 4.1)
        
        conv_out = self.resnet(image)  # (batch_size, 2048, image_size/32, image_size/32)
        conv_out = F.normalize(conv_out, p=2, dim=1)
        
        #qn = torch.norm(conv_out, p=2, dim=1, keepdim=True)#.detach()
        #conv_out = conv_out.div(qn.expand_as(conv_out))

        #Chargrid here and 2 * 64
        if self.chargrid_channels > 0:
            chargrid = F.normalize(chargrid, p=2, dim=1)
            chargrid = self.chargrid_net(chargrid)
            conv_out = torch.cat([conv_out,chargrid],1)
            conv_out = self.entitygrid_net(conv_out)

        # normalize by feature map, why need it and why not??
        # conv_out = conv_out / (conv_out.norm(p=2, dim=1, keepdim=True).expand_as(
        #     conv_out) + 1e-8)  # Section 3.1 of show, ask, attend, tell

        # final_out = self.mlp(
        #     conv_out.reshape(conv_out.size(0), -1))  # (batch_size , 2048*14*14) -> (batch_size, n_class)
        # conv_out = conv_out.view(conv_out.size(0), -1).contiguous()

        embed = self.embed(question)
        embed_pack = nn.utils.rnn.pack_padded_sequence(
            embed, question_len, batch_first=True
        )
        lstm_output, (h, c) = self.lstm(embed_pack)

        # pad packed sequence to get last timestamp of lstm hidden
        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_output, batch_first=True)

        idx = (torch.LongTensor(question_len) - 1).view(-1, 1).expand(
            len(question_len), lstm_output.size(2))
        time_dimension = 1  # if batch_first else 0
        idx = idx.unsqueeze(time_dimension).to(device)

        lstm_final_output = lstm_output.gather(
            time_dimension, idx).squeeze(time_dimension)

        attention = self.attention(conv_out, lstm_final_output)
        weighted_conv_out = apply_attention(conv_out,
                                            attention)  # (n, glimpses * channel) ## Or should be (n, glimpses * channel, H, W)?
        # augmented_lstm_output = (weighted_conv_out + lstm_final_output)
        augmented_lstm_output = torch.cat((weighted_conv_out, lstm_final_output), 1)


        if self.hop == 2:
            raise NotImplementedError
            # attention = self.attention(conv_out, lstm_final_output)
            # weighted_conv_out = apply_attention(conv_out, attention)
            # augmented_lstm_output = (weighted_conv_out + lstm_final_output)

        

        return self.mlp(augmented_lstm_output)


    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        if (not fine_tune):
            for p in self.resnet.parameters():
                p.requires_grad = False
        if (not load_from_hdf5):
            for p in self.resnet.parameters():
                p.requires_grad = False
        else:
            #for c in list(self.resnet.children())[:5]:
            #    for p in c.parameters():
            #        p.requires_grad = False
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune
            if self.chargrid_channels > 0:
                for c in list(self.chargrid_net.children()):
                    for p in c.parameters():
                        p.requires_grad = fine_tune
                

        for p in self.mlp.parameters():
            p.requires_grad = True

        for p in self.attention.parameters():
            print(p.requires_grad, "p.requires_grad")
            p.requires_grad = True