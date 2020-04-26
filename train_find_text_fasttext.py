# -*- coding: utf-8 -*-
import ast
import sys
import pickle
from collections import Counter

import comet_ml

import torch
import torchvision
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from dataset_find_text_fasttext import DVQA, collate_data, transform
import time
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import textwrap
import seaborn as sns

sys.path.append('/workspace/st_vqa_entitygrid/solution/')
#sys.path.append('/project/paul_op_masterthesis/st_vqa_entitygrid/solution/')
from dvqa import enlarge_batch_tensor
from visualize import TensorBoardVisualize,SaveFeatures
from sklearn.metrics import precision_recall_fscore_support

from model_find_text_fasttext import SANVQA


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
n_epoch = 4000
reverse_question = False
batch_size = 6#(64 if model_name == "QUES" else 32) if torch.cuda.is_available() else 4
n_workers = 0 #0  # 4
clip_norm = 50
load_image = False

#Saving Parameters (every 
saving_epoch = 1
train_progress_iteration = 5
train_visualization_iteration = 100
validation_epoch = 1

n_label_channels = 41

def train(epoch,tensorboard_client,global_iteration,word_dic,answer_dic,load_image=True, model_name=None):

    run_name = "train"
    model.train(True)  # train mode
    if isinstance(model,nn.DataParallel):
        vqa_model = model.module
    else:
        vqa_model = model

    dataset = iter(train_set)
    #Debug
    pbar = tqdm(dataset)
    #pbar = dataset
    n_batch = len(pbar)
    moving_loss = 0  # it will change when loop over data

    print(device)
    #print(next(model.parameters()).is_cuda, "next(model.parameters()).is_cuda")


    
#Chargrid load labels,bboxes
##    for i, (image, question, q_len, answer, question_class, bboxes, n_bboxes, data_index) in enumerate(pbar):
    for i, (image, question, q_len, answer, question_class, embeddings,bboxes,emb_lengths, data_index) in enumerate(pbar):
        tensorboard_client.set_epoch_step(epoch,global_iteration)

        image, question, q_len, answer, embeddings = (##bboxes = (
            image.to(device),
            question.to(device),
            torch.tensor(q_len),
            answer.to(device),
            ##bboxes.to(device)
            embeddings.to(device)
        )

        tmp_batch_size = question.shape[0]

        #Train: Batch Loop
        wordgrid = torch.zeros((tmp_batch_size,300,224,224),device=device)
        for batch_i in range(tmp_batch_size):
            for emb_i in range(embeddings.size(1)):
                x,y,x2,y2 = bboxes[batch_i,emb_i,:]
                emb_box = embeddings[batch_i,emb_i].repeat((x2-x,y2-y,1)).transpose(2,0)
                wordgrid[batch_i,:,y:y2,x:x2] = emb_box

        wordgrid = F.normalize(wordgrid,p=2,dim=1)

        #One Hot Encoding of Words
        ##encoded_chargrid = torch.zeros((tmp_batch_size, 1200,224,224),device=device)
        ##encoded_chargrid = encoded_chargrid.scatter_(1, chargrid.unsqueeze(1), 1)
        ##encoded_chargrid[:,0,:,:] = 0. #label at index 0 is " "

        #One Hot Encoding of Word in question
        ##encoded_question = torch.zeros((tmp_batch_size, 1200),device=device)
        ##encoded_question = encoded_question.scatter_(1, question, 1)
        
        model.zero_grad()
        output = model(image, question, q_len, wordgrid)

        loss = torch.mean((output == (answer == 6.0)).float())
        #loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        #optimizer.step()

        visualize_train(
            global_iteration,run_name,tensorboard_client,
            loss)

        global_iteration += 1

    #valid(epoch + float(i * batch_size / 2325316),tensorboard_client,global_iteration, train_set, model_name=model_name,
    #            load_image=load_image, val_split="train")
    
    return global_iteration,moving_loss
    

def visualize_train(global_iteration,run_name,tensorboard_client,loss):
    if (global_iteration % train_progress_iteration != 0):
        return

    
    # - loss
    loss_dic = {"loss":loss.detach().item()}
    tensorboard_client.append_line(global_iteration,loss_dic,"Training/running_loss")
    
    with tensorboard_client.comet_exp.train():
        tensorboard_client.comet_line(loss_dic,"running")


def valid(epoch,tensorboard_client,global_iteration, valid_set, load_image=True, model_name=None, val_split="val_easy"):
    #run_name = val_split
    
    print("Inside validation ", epoch)
    dataset = iter(valid_set)
    model.eval()  # eval_mode
    class_correct = Counter()
    class_total = Counter()
    losses,prediction,all_output_count,correct_output_count,all_answer_count = [],[],Counter(),Counter(),Counter()
    
    with torch.no_grad():

        ##for i, (image, question, q_len, answer, answer_class, bboxes, n_bboxes, data_index) in enumerate(tqdm(dataset)):
        for i, (image, question, q_len, answer, answer_class, chargrid, data_index) in enumerate(tqdm(dataset)):

            image, question, q_len, chargrid = (
                image.to(device),
                question.to(device),
                torch.tensor(q_len),
                ##bboxes.to(device)
                chargrid.to(device)
            )

            #batch_size = labels.shape[0]
            #n_channel = labels.shape[-1]
            tmp_batch_size = question.shape[0]
            ##chargrid = chargrid_creation(bboxes,n_bboxes,question.get_device(),tmp_batch_size)
            encoded_chargrid = torch.zeros((tmp_batch_size, n_label_channels,224,224),device=device)
            encoded_chargrid = encoded_chargrid.scatter_(1, chargrid.unsqueeze(1), 1)
            #Chargrid Creation 
            # for batch_id in range(labels.shape[0]):
            #     for label_id in range(n_label[batch_id].item()):
            #         x,y,x2,y2 = bboxes[batch_id,label_id,:]
            #         label_box = labels[batch_id,label_id].repeat((x2-x,y2-y,1)).transpose(2,0)
            #         chargrid[batch_id,:,y:y2,x:x2] = label_box

            

            output = model(image, question, q_len, encoded_chargrid)
            cpu_output = output.data.cpu()
            loss = criterion(cpu_output, answer)
            losses.append(loss.item())

            argmax_output = cpu_output.numpy().argmax(1)
            numpy_answer = answer.numpy()
            
            correct = argmax_output == numpy_answer
            for c, class_ in zip(correct, answer_class):
                if c:  # if correct
                    class_correct[class_] += 1
                class_total[class_] += 1

            prediction.append([data_index.numpy(),numpy_answer,argmax_output])

            #all_output_count.update(argmax_output)
            #correct_output_count.update(argmax_output[correct])
            #all_answer_count.update(numpy_answer)

            if (("IMG" in model_name) or ("SAN" in model_name)) and type(epoch) == type(0.1) and (
                    i * batch_size // 2) > (
                    6e4):  # intermediate train, only val on 10% of the validation set
                break  # early break validation loop

    class_correct['total'] = sum(class_correct.values())
    class_total['total'] = sum(class_total.values())

    prediction = np.concatenate(prediction,axis=1)
    

    
    

    #Debug
    # with open('log/log_' + model_name + '_{}_'.format(round(epoch + 1, 4)) + val_split + '.txt', 'w') as w:
    #     for k, v in class_total.items():
    #         w.write('{}: {:.5f}\n'.format(k, class_correct[k] / v))
    #     # TODO: save the model here!

    total_score = class_correct['total'] / class_total['total']
    print('Avg Acc: {:.5f}'.format(total_score))

    visualize_val(
        global_iteration,tensorboard_client,val_split,len(dataset),epoch,
        class_total,class_correct,losses,
        prediction)

    return prediction,total_score

def visualize_val(
    global_iteration,tensorboard_client,val_split,n_batch,epoch,
    class_total,class_correct,losses,
    prediction):

    correct_pred = prediction[1] == prediction[2]

    all_output_count = Counter(prediction[2])
    correct_output_count = Counter(prediction[2][correct_pred])
    all_answer_count = Counter(prediction[1])
    average_loss = sum(losses) / n_batch

    #Precision, Recall
    unique_answers = list(Counter(all_answer_count).keys())
    precision,recall,f1_score,supp = precision_recall_fscore_support(prediction[1],prediction[2],labels=unique_answers)


    chart_dic = {k:class_correct[k] / v
        for k, v in class_total.items()
    }
    tensorboard_client.append_line(
        global_iteration,chart_dic,
            f"Evaluation/{val_split}_accuracy")

    tensorboard_client.append_line(
        global_iteration,{"average_loss":average_loss},
            f"Evaluation/{val_split}_avg_loss")

    prec_dic = {
            f"prec_{answer}":prec 
            for prec,answer in zip(*[precision,unique_answers])}

    tensorboard_client.append_line(
        global_iteration,prec_dic,
            f"Evaluation/{val_split}_precision")

    rec_dic = {
            f"rec_{answer}":rec 
            for rec,answer in zip(*[recall,unique_answers])}

    tensorboard_client.append_line(
        global_iteration,rec_dic,
            f"Evaluation/{val_split}_recall")

    f1_dic = {
            f"f1_{answer}":f1 
            for f1,answer in zip(*[f1_score,unique_answers])}

    tensorboard_client.append_line(
        global_iteration,f1_dic,
            f"Evaluation/{val_split}_f1_score")

    with tensorboard_client.comet_exp.test():
        [chart_dic.update(dic) for dic in [prec_dic,rec_dic,rec_dic,f1_dic,{"average_loss":average_loss}]]
        tensorboard_client.comet_line(chart_dic,f"{val_split}")
        


    print("class_correct", class_correct)
    print("class_total", class_total)
    print("all_output_count", all_output_count)
    print("correct_output_count", correct_output_count)
    print("average_loss", average_loss )
    print("f1", f1_dic )
    print("recall", rec_dic )
    print("precision", prec_dic )
    
    

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

    if model_name == "SANVQA" or "SANVQAbeta":
        model = SANVQA(n_answers, n_vocab=n_words, encoded_image_size=(14 if load_from_hdf5 == False else 7))
        #model = nn.DataParallel(model)
        load_image = True
        
        model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma) # Decays the learning rate of each parameter group by gamma every step_size epochs.

    train_set = DataLoader(
        DVQA(
            sys.argv[1],
            transform=transform,
            reverse_question=reverse_question,
            use_preprocessed=True,
            load_image=load_image,
            load_from_hdf5=load_from_hdf5,
            file_name=sys.argv[5]

        ),
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=True,
        collate_fn=collate_data,
    )

    #Debug
    valid_set_easy = DataLoader(
        DVQA(
            sys.argv[1],
            "val_easy",
            transform=None,
            reverse_question=reverse_question,
            use_preprocessed=True,
            load_image=load_image,
            load_from_hdf5=load_from_hdf5,
            file_name=sys.argv[6]
        ),
        batch_size=batch_size,# // 2,
        num_workers=n_workers,
        collate_fn=collate_data,  ## shuffle=False

    )
    """
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
    """

    #Chargrid: Create Tensorboard Writer
    tensorboard_client = TensorBoardVisualize(sys.argv[3],"log/",dic)
    tensorboard_client.comet_exp.set_model_graph(str(model))
    global_iteration = 0
    start_epoch = 0

    global_iteration = len(train_set)*start_epoch
    for epoch in range(start_epoch,n_epoch+start_epoch):
        tensorboard_client.set_epoch_step(epoch,global_iteration)
        # if scheduler.get_lr()[0] < lr_max:
        #     scheduler.step()
        print("epoch=", epoch)

        # TODO: add load model from checkpoint
        checkpoint_name = 'checkpoint/checkpoint_' + sys.argv[3] + '_{}.model'.format(str(epoch + 1).zfill(3))
        #if os.path.exists(checkpoint_name):
        #     # model.load_state_dict(torch.load(model.state_dict())
        #     model.load_state_dict(
        #         torch.load(checkpoint_name, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        #     continue
        global_iteration,moving_loss = train(
            epoch,
            tensorboard_client,
            global_iteration,
            train_set.dataset.word_class,
            train_set.dataset.answer_class,
            load_image=load_image, model_name=model_name)
        #Debug        
        tensorboard_client.close()
        


