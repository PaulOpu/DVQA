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

#sys.path.append('/workspace/st_vqa_entitygrid/solution/')
sys.path.append('/project/paul_op_masterthesis/st_vqa_entitygrid/solution/')
from dvqa import enlarge_batch_tensor
from visualize import TensorBoardVisualize



# if torch.__version__ == '1.1.0':
#     from torchvision.models.resnet import resnet101 as _resnet101
# else:
from torchvision.models import resnet101 as _resnet101

# from torchvision.models import resnet152 as _resnet152

# from model import RelationNetworks

model_name = "SANVQA"  # "SANVQAbeta" # "SANVQA"  # "IMGQUES"  # "IMG"  # "IMG"  # "QUES"  # "YES"
use_annotation = True if model_name == "SANDY" else False
lr = 1e-3
lr_max = 1e-2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_parallel = False
lr_step = 20
lr_gamma = 2  # gamma (float) – Multiplicative factor of learning rate decay. Default: 0.1.
weight_decay = 1e-4
n_epoch = 5
reverse_question = False
batch_size = (64 if model_name == "QUES" else 32) if torch.cuda.is_available() else 4
n_workers = 0 #0  # 4
clip_norm = 50
load_image = False




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
        self.n_class = n_class
        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.lstm = nn.LSTM(embed_hidden, lstm_hidden, batch_first=True)

        self.enc_image_size = encoded_image_size
        # resnet = torchvision.models.resnet152(
        #     pretrained=True)

        
          # 051019, batch_size changed to 32 from 64. ResNet 152 to Resnet 101.
        # pretrained ImageNet ResNet-101, use the output of final convolutional layer after pooling

        
        self.dropout = nn.Dropout(
            p=dropout_rate)  # insert Dropout layers after convolutional layers that you’re going to fine-tune

        self.attention = Attention(conv_output_size, lstm_hidden, mid_features=512, glimpses=glimpses, drop=0.5)

        self.mlp = nn.Sequential(self.dropout,  ## TODO: shall we eliminate this??
                                 nn.Linear(conv_output_size * glimpses + lstm_hidden,
                                           mlp_hidden_size),
                                 nn.ReLU(),
                                 self.dropout,
                                 nn.Linear(mlp_hidden_size, self.n_class))

          # define which parameter sets are to be fine-tuned
        self.hop = 1

        act_f = nn.ReLU()

        resnet = torchvision.models.resnet101(
            pretrained=True)
            
        #modules = list(resnet.children())[:-2]
        #EarlyConcat: modules = list(resnet.children())[:5]
        concat_point = 6
        modules = list(resnet.children())[:concat_point]

        #self.resnet = nn.Sequential(*modules)
        self.resnet = nn.Sequential(*modules)

        #Chargrid: Early Concat
        untrained_resnet = torchvision.models.resnet101(
            pretrained=False)

        #EarlyConcat: modules = list(untrained_resnet.children())[:5]
        chargrid_modules = list(untrained_resnet.children())[:concat_point]
        #replace first module to fit dimensions
        chargrid_modules[0:1] = [
            nn.Conv2d(45, 10, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(10),
            act_f,
            nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3)]

        self.chargrid_net = nn.Sequential(*chargrid_modules)

        #After Concat

        #EarlyConcat: entitygrid_modules = list(untrained_resnet.children())[5:-2]
        entitygrid_modules = list(untrained_resnet.children())[concat_point:-2]
        #concatentation doubles the number of channels (512->1024)
        entitygrid_modules[0][0].conv1 = nn.Conv2d(
            1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        
        entitygrid_modules[0][0].downsample[0] = nn.Conv2d(
            1024, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)

        self.entitygrid_net = nn.Sequential(*entitygrid_modules)
        
        
        self.fine_tune(True)

        #Chargrid: Network before concat with image (224/112/56/28/14)
        """ self.chargrid_net = nn.Sequential(
                nn.Conv2d(45, 10, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(10),
                act_f,
                nn.Conv2d(10, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                act_f,
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                act_f,
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                act_f,
                nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(1024),
                act_f
            )

        self.entitygrid_net = nn.Sequential(
                #nn.Conv2d(3072, 2048, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(2048),
                #act_f,
                nn.Conv2d(3072, 2048, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(2048),
                act_f,
                nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1024),
                act_f,
                nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(2048),
                act_f,
                nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(2048),
                act_f
            ) """




    def forward(self, image, question, question_len, chargrid):  # this is an image blind example (as in section 4.1)
        #conv_out = self.resnet(image)  # (batch_size, 2048, image_size/32, image_size/32)
        conv_out = self.resnet(image)

        # normalize by feature map, why need it and why not??
        # conv_out = conv_out / (conv_out.norm(p=2, dim=1, keepdim=True).expand_as(
        #     conv_out) + 1e-8)  # Section 3.1 of show, ask, attend, tell

        # final_out = self.mlp(
        #     conv_out.reshape(conv_out.size(0), -1))  # (batch_size , 2048*14*14) -> (batch_size, n_class)
        # conv_out = conv_out.view(conv_out.size(0), -1).contiguous()

        #Chargrid: Enlarge Img Vector
        #conv_out = enlarge_batch_tensor(conv_out)
        chargrid = self.chargrid_net(chargrid)

        conv_out = torch.cat([conv_out,chargrid],dim=1)
        conv_out = self.entitygrid_net(conv_out)

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
            for c in list(self.resnet.children()):
            #for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune
            

        for p in self.mlp.parameters():
            p.requires_grad = True

        for p in self.attention.parameters():
            print(p.requires_grad, "p.requires_grad")
            p.requires_grad = True

        #Chargrid and EntityGrid
        for c in list(self.chargrid_net.children()):
            for p in c.parameters():
                p.requires_grad = True
        for c in list(self.entitygrid_net.children()):
            for p in c.parameters():
                p.requires_grad = True



def train(epoch,tensorboard_client,global_iteration, load_image=True, model_name=None):
    run_name = "train"
    model.train(True)  # train mode

    dataset = iter(train_set)
    #pbar = tqdm(dataset)
    pbar = dataset
    n_batch = len(pbar)
    moving_loss = 0  # it will change when loop over data



    #Train Epoch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    print(device)
    print(next(model.parameters()).is_cuda, "next(model.parameters()).is_cuda")
    #Chargrid load labels,bboxes
    for i, (image, question, q_len, answer, question_class, labels, bboxes, n_label) in enumerate(pbar):
        start.record()
        image, question, q_len, answer,labels,bboxes,n_label = (
            image.to(device),
            question.to(device),
            torch.tensor(q_len),
            answer.to(device),
            labels.to(device),
            bboxes,#bboxes.to(device),
            n_label
        )
        #end.record()
        #torch.cuda.synchronize()
        #print("Loading: ",start.elapsed_time(end))
        #start.record()
        #Chargrid: Creation
        batch_size = labels.shape[0]
        n_channel = labels.shape[-1]
        chargrid = torch.zeros((batch_size,n_channel,224,224),device=torch.get_device(labels))
        #create chargrid on the fly
        #start = time.time()
        for batch_id in range(labels.shape[0]):
            for label_id in range(n_label[batch_id].item()):
                x,y,x2,y2 = bboxes[batch_id,label_id,:]
                label_box = labels[batch_id,label_id].repeat((x2-x,y2-y,1)).transpose(2,0)
                chargrid[batch_id,:,y:y2,x:x2] = label_box

        #end.record()
        #torch.cuda.synchronize()
        #print("Chargrid: ",start.elapsed_time(end))
        #start.record()
        #for batch_id in range(batch_size):
        #    for label_id in range(n_label[batch_id].item()):
        #        x,y,x2,y2 = bboxes[batch_id,label_id,:]
        #        chargrid[batch_id,y:y2,x:x2,:] = labels[batch_id,label_id]
        #print(f"chargrid: {time.time()-start:.4f}",)
        #chargrid = chargrid.permute(0,3,1,2)

        model.zero_grad()
        output = model(image, question, q_len, chargrid)
        #end.record()
        #torch.cuda.synchronize()
        #print("Forward: ",start.elapsed_time(end))
        #start.record()
        #SANDY: add the OCR tokens at the beginning of the question
        loss = criterion(output, answer)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        #end.record()
        #torch.cuda.synchronize()
        #print("Backward: ",start.elapsed_time(end))
        #start.record()
        correct = output.data.cpu().numpy().argmax(1) == answer.data.cpu().numpy()
        correct = correct.sum() / batch_size

        if moving_loss == 0:
            moving_loss = correct
            # print("moving_loss = correct")

        else:
            moving_loss = moving_loss * 0.99 + correct * 0.01
            # print("moving_loss = moving_loss * 0.99 + correct * 0.01")

        #pbar.set_description(
        #    'Epoch: {}; Loss: {:.5f}; Acc: {:.5f}; Correct:{:.5f}; LR: {:.6f}'.format(
        #        epoch + 1,
        #        loss.detach().item(),  # 0.00 for YES model
        #        moving_loss,
        #        correct,
        #        optimizer.param_groups[0]['lr'],  # 0.00  for YES model
        #    )
        #)
        if (("IMG" in model_name) or ("SAN" in model_name)) and i % 10000 == 0 and i != 0:
            # valid(epoch + float(i * batch_size / 2325316), model_name=model_name, val_split="val_easy",
            #       load_image=load_image)
            valid(epoch + float(i * batch_size / 2325316),tensorboard_client,global_iteration, valid_set_easy, model_name=model_name,
                  load_image=load_image, val_split="val_easy")

            model.train(True)

        #Chargrid: visualize
        visualize_train(global_iteration,run_name,model,tensorboard_client,loss,moving_loss)
        #end.record()
        #torch.cuda.synchronize()
        #print("Visu: ",start.elapsed_time(end))

        global_iteration += 1
        # end = time.time()
        # print("finished accuracy calculation", end - start)
        # start = time.time()

    return global_iteration  


def visualize_weight_gradient(global_iteration,tensorboard_client,module,weight_name,gradient_name):

    weights = module.weight.data.cpu().numpy()
    gradients = module.weight.grad.cpu().numpy()
    tensorboard_client.append_histogram(global_iteration, weights.reshape(-1), weight_name)
    tensorboard_client.append_histogram(global_iteration, gradients.reshape(-1), gradient_name)


def visualize_train(global_iteration,run_name,model,tensorboard_client,loss,moving_loss,**kwargs):
    if global_iteration % 100 != 0:
        return
    # (1-1)*3000+100
    # - weights

    # - gradients
    name = "Chargrid"
    chargrid_net = model.chargrid_net

    visualize_weight_gradient(
        global_iteration,
        tensorboard_client,
        chargrid_net[0],
        f"{name}_weights/embedding",
        f"{name}_gradients/embedding"
    )

    visualize_weight_gradient(
        global_iteration,
        tensorboard_client,
        chargrid_net[3],
        f"{name}_weights/resnet_conv1",
        f"{name}_gradients/resnet_conv1"
    )

    #Chargrid: Early Concat
    for bottlenet_id,bottleneck in enumerate(chargrid_net[-1]):
        for conv_name in ["conv1"]:
            visualize_weight_gradient(
                global_iteration,
                tensorboard_client,
                getattr(bottleneck,conv_name),
                f"{name}_weights/conv2_bottlenet{bottlenet_id}_{conv_name}",
                f"{name}_gradients/conv2_bottlenet{bottlenet_id}_{conv_name}"
            )

    #EntityGrid Net
    name = "Entitygrid"
    entitygrid_net = model.entitygrid_net
    """ for idx in [0,3,6,9]:
        weights = entitygrid_net[idx].weight.data.cpu().numpy()
        gradients = entitygrid_net[idx].weight.grad.cpu().numpy()
        tensorboard_client.append_histogram(global_iteration, weights.reshape(-1), f"{name}/weights_{idx}")
        tensorboard_client.append_histogram(global_iteration, gradients.reshape(-1), f"{name}/gradients_{idx}")
    """
    #Chargrid: Early Concat
    for bottlenet_id,bottleneck in enumerate(entitygrid_net[0]):
        for conv_name in ["conv1"]:
            visualize_weight_gradient(
                global_iteration,
                tensorboard_client,
                getattr(bottleneck,conv_name),
                f"{name}_weights/conv3_bottlenet{bottlenet_id}_{conv_name}",
                f"{name}_gradients/conv3_bottlenet{bottlenet_id}_{conv_name}"
            )

    for bottlenet_id,bottleneck in enumerate(entitygrid_net[-1]):
        for conv_name in ["conv1"]:
            visualize_weight_gradient(
                global_iteration,
                tensorboard_client,
                getattr(bottleneck,conv_name),
                f"{name}_weights/conv5_bottlenet{bottlenet_id}_{conv_name}",
                f"{name}_gradients/conv5_bottlenet{bottlenet_id}_{conv_name}"
            )
    
    # - loss
    tensorboard_client.append_line(global_iteration,{"loss":loss.detach().item()},"Metrics/running_loss")
    # - accuracy
    tensorboard_client.append_line(global_iteration,{run_name:moving_loss},"Metrics/accuracy")

    tensorboard_client.close()


def valid(epoch,tensorboard_client,global_iteration, valid_set, load_image=True, model_name=None, val_split="val_easy"):
    run_name = val_split
    
    print("Inside validation ", epoch)
    # valid_set = DataLoader(
    #     DVQA(
    #         sys.argv[1],
    #         val_split,
    #         transform=None,
    #         reverse_question=reverse_question,
    #         use_preprocessed=True,
    #         load_image=load_image,
    #         load_from_hdf5=load_from_hdf5
    #     ),
    #     batch_size=batch_size // 2,
    #     num_workers=4,
    #     collate_fn=collate_data,  ## shuffle=False
    #
    # )
    # valid_set=valid_set_easy if val_split=="val_easy" else val
    dataset = iter(valid_set)
    model.eval()  # eval_mode
    class_correct = Counter()
    class_total = Counter()

    with torch.no_grad():
        for i, (image, question, q_len, answer, answer_class, labels, bboxes, n_label) in enumerate(tqdm(dataset)):
            image, question, q_len, labels, bboxes, n_label = (
                image.to(device),
                question.to(device),
                torch.tensor(q_len),
                labels.to(device),
                bboxes,#bboxes.to(device),
                n_label
            )

            batch_size = labels.shape[0]
            n_channel = labels.shape[-1]
            chargrid = torch.zeros((batch_size,n_channel,224,224),device=torch.get_device(labels))

            #Chargrid Creation 
            for batch_id in range(labels.shape[0]):
                for label_id in range(n_label[batch_id].item()):
                    x,y,x2,y2 = bboxes[batch_id,label_id,:]
                    label_box = labels[batch_id,label_id].repeat((x2-x,y2-y,1)).transpose(2,0)
                    chargrid[batch_id,:,y:y2,x:x2] = label_box
            #for batch_id in range(batch_size):
            #    for label_id in range(n_label[batch_id].item()):
            #        x,y,x2,y2 = bboxes[batch_id,label_id,:]
            #        chargrid[batch_id,y:y2,x:x2,:] = labels[batch_id,label_id]
            #chargrid = chargrid.permute(0,3,1,2)

            output = model(image, question, q_len, chargrid)
            correct = output.data.cpu().numpy().argmax(1) == answer.numpy()
            for c, class_ in zip(correct, answer_class):
                if c:  # if correct
                    class_correct[class_] += 1
                class_total[class_] += 1

            if (("IMG" in model_name) or ("SAN" in model_name)) and type(epoch) == type(0.1) and (
                    i * batch_size // 2) > (
                    6e4):  # intermediate train, only val on 10% of the validation set
                break  # early break validation loop

            

    class_correct['total'] = sum(class_correct.values())
    class_total['total'] = sum(class_total.values())

    print("class_correct", class_correct)
    print("class_total", class_total)

    with open('log/log_' + model_name + '_{}_'.format(round(epoch + 1, 4)) + val_split + '.txt', 'w') as w:
        for k, v in class_total.items():
            w.write('{}: {:.5f}\n'.format(k, class_correct[k] / v))
        # TODO: save the model here!

    print('Avg Acc: {:.5f}'.format(class_correct['total'] / class_total['total']))

    visualize_val(global_iteration,run_name,model,tensorboard_client,val_split,class_total,class_correct)

def visualize_val(global_iteration,run_name,model,tensorboard_client,val_split,class_total,class_correct):
    chart_dic = {k:class_correct[k] / v
        for k, v in class_total.items()
    }
    tensorboard_client.append_line(
        global_iteration,chart_dic,
            f"Metrics/{val_split}_accuracy")

    tensorboard_client.close()

if __name__ == '__main__':
    data_path = sys.argv[1]
    with open(os.path.join(data_path,'dic.pkl'), 'rb') as f:
        dic = pickle.load(f)

    n_words = len(dic['word_dic']) + 1
    n_answers = len(dic['answer_dic'])
    global load_from_hdf5
    load_from_hdf5 = ast.literal_eval(sys.argv[2])
    print("load_from_hdf5", load_from_hdf5)
    print(n_answers, "n_answers")

    if model_name == "YES":
        yes_answer_idx = dic['answer_dic']["yes"]
        model = YES(n_answers, yes_answer_idx)
        load_image = False

    elif model_name == "IMG":
        # load_from_hdf5 = False
        model = IMG(n_answers, encoded_image_size=(14 if load_from_hdf5 == False else 7))
        # if data_parallel:
        #     model = nn.DataParallel(model)
        load_image = True
        model = model.to(device)
    elif model_name == "QUES":
        load_from_hdf5 = False
        load_image = False
        model = QUES(n_answers, n_vocab=n_words)
        model = model.to(device)

    elif model_name == "IMGQUES":
        model = IMGQUES(n_answers, n_vocab=n_words, encoded_image_size=(14 if load_from_hdf5 == False else 7))
        load_image = True
        model = model.to(device)

    elif model_name == "SANVQA" or "SANVQAbeta":
        model = SANVQA(n_answers, n_vocab=n_words, encoded_image_size=(14 if load_from_hdf5 == False else 7))
        load_image = True
        model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma) # Decays the learning rate of each parameter group by gamma every step_size epochs.

    train_set = DataLoader(
        DVQA(
            sys.argv[1],
            transform=transform,
            reverse_question=reverse_question,
            use_preprocessed=True,
            load_image=load_image,
            load_from_hdf5=load_from_hdf5

        ),
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=True,
        collate_fn=collate_data,
    )

    valid_set_easy = DataLoader(
        DVQA(
            sys.argv[1],
            "val_easy",
            transform=None,
            reverse_question=reverse_question,
            use_preprocessed=True,
            load_image=load_image,
            load_from_hdf5=load_from_hdf5
        ),
        batch_size=batch_size // 2,
        num_workers=n_workers,
        collate_fn=collate_data,  ## shuffle=False

    )

    valid_set_hard = DataLoader(
        DVQA(
            sys.argv[1],
            "val_hard",
            transform=None,
            reverse_question=reverse_question,
            use_preprocessed=True,
            load_image=load_image,
            load_from_hdf5=load_from_hdf5
        ),
        batch_size=batch_size // 2,
        num_workers=n_workers,
        collate_fn=collate_data,  ## shuffle=False

    )

    #Chargrid: Create Tensorboard Writer
    tensorboard_client = TensorBoardVisualize(sys.argv[3],"log/")
    global global_iteration
    global_iteration = 0

    #Chargrid: Continue from Checkpoint
    checkpoint_epoch = int(sys.argv[4])
    start_epoch = 0
    if checkpoint_epoch > 0:
        checkpoint_name = 'checkpoint/checkpoint_' + model_name + '_{}.model'.format(str(checkpoint_epoch).zfill(3))
        #model.load_state_dict(torch.load(model.state_dict())
        model.load_state_dict(
            torch.load(checkpoint_name, map_location='cuda'))
        global_iteration = len(train_set)
        start_epoch = checkpoint_epoch

    for epoch in range(start_epoch,n_epoch):
        # if scheduler.get_lr()[0] < lr_max:
        #     scheduler.step()
        print("epoch=", epoch)

        # TODO: add load model from checkpoint
        checkpoint_name = 'checkpoint/checkpoint_' + model_name + '_{}.model'.format(str(epoch + 1).zfill(3))
        #if os.path.exists(checkpoint_name):
        #     # model.load_state_dict(torch.load(model.state_dict())
        #     model.load_state_dict(
        #         torch.load(checkpoint_name, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        #     continue
        global_iteration = train(epoch,tensorboard_client,global_iteration, load_image=load_image, model_name=model_name)
        valid(epoch,tensorboard_client,global_iteration, valid_set_easy, model_name=model_name, load_image=load_image, val_split="val_easy")
        valid(epoch,tensorboard_client,global_iteration, valid_set_hard, model_name=model_name, load_image=load_image, val_split="val_hard")
        with open(checkpoint_name, 'wb'
                  ) as f:
            torch.save(model.state_dict(), f)

        print("model saved! epoch=", epoch)
