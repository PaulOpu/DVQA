# -*- coding: utf-8 -*-
import ast
import sys
import pickle
import json
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
from dataset_find_text_transf2 import DVQA, collate_data, transform
import time
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import textwrap
import seaborn as sns
from sklearn.metrics import mean_squared_error

sys.path.append('/workspace/st_vqa_entitygrid/solution/')
#sys.path.append('/project/paul_op_masterthesis/st_vqa_entitygrid/solution/')
from dvqa import enlarge_batch_tensor
from visualize import TensorBoardVisualize,SaveFeatures
from sklearn.metrics import precision_recall_fscore_support
from torch.nn import CosineSimilarity
from model_find_text_transf2 import Encoder


# if torch.__version__ == '1.1.0':
#     from torchvision.models.resnet import resnet101 as _resnet101
# else:
from torchvision.models import resnet101 as _resnet101

# from torchvision.models import resnet152 as _resnet152

# from model import RelationNetworks

# model_name = "SANVQA"  # "SANVQAbeta" # "SANVQA"  # "IMGQUES"  # "IMG"  # "IMG"  # "QUES"  # "YES"
# use_annotation = True if model_name == "SANDY" else False
# lr = 1e-3
# lr_max = 1e-3
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data_parallel = False
# lr_step = 20
# lr_gamma = 2  # gamma (float) – Multiplicative factor of learning rate decay. Default: 0.1.
# weight_decay = 1e-4
# n_epoch = 4000
# reverse_question = False
# batch_size = 16#(64 if model_name == "QUES" else 32) if torch.cuda.is_available() else 4
# n_workers = 0 #0  # 4
# clip_norm = 50
# load_image = False

# #Saving Parameters (every 
# saving_epoch = 1
# train_progress_iteration = 5
# train_visualization_iteration = 50
# validation_epoch = 1

# n_label_channels = 41

def train(epoch,tensorboard_client,global_iteration,train_set,word_dic,answer_dic,load_image=True, model_name=None):
    run_name = "train"
    model.train(True)

    
      # train mode
    if isinstance(model,nn.DataParallel):
        vqa_model = model.module
    else:
        vqa_model = model

    #att_module = vqa_model.layers[0].multi_head_attention

    dataset = iter(train_set)
    #Debug
    pbar = tqdm(dataset)
    #pbar = dataset
    n_batch = len(pbar)
    moving_loss = 0  # it will change when loop over data

    answer_dic = {
        "yes":[],
        "no":[]
        }
    
#Chargrid load labels,bboxes
##    for i, (image, question, q_len, answer, question_class, bboxes, n_bboxes, data_index) in enumerate(pbar):
    for i, (image, question_idx, question, q_len, answer, question_class, embeddings,bboxes,embedding_ids,emb_lengths, data_index) in enumerate(pbar):
        tensorboard_client.set_epoch_step(epoch,global_iteration)

        image, question, question_idx, q_len, answer, bboxes, embeddings, emb_lengths, embedding_ids = (
            image.to(device),
            question.to(device),
            question_idx.to(device),
            torch.tensor(q_len).to(device),
            answer.to(device),
            bboxes.to(device),
            embeddings.to(device),
            emb_lengths.to(device),
            embedding_ids.to(device)
        )

        #Train: Batch Loop
        
        vqa_model.zero_grad()
        #output,question,wordgrid = vqa_model(image, question, q_len, embeddings, bboxes, emb_lengths)

        prediction,att_vector,question = vqa_model(
            image, question, question_idx, q_len, embeddings, bboxes, embedding_ids, emb_lengths)

        label = answer.clone()
        #label[label == 6.0] = 1.
        #label[label == 7.0] = -1.
        label[label == 6.0] = 1
        label[label == 7.0] = 0

        prediction = prediction.squeeze()
        #loss = criterion(question,att_question, label.float())
        loss = criterion(prediction, label.float())
        loss.backward()
        optimizer.step()

        
        loss = loss.detach()
        #correct = prediction.argmax(1) == label
        correct = (prediction.squeeze() > 0.5) == label
        #correct = (cos_similarity(att_question,question) > 0.5) == (answer == 6.)
        mean_correct = (correct.sum().float() / batch_size).detach()
        

        if moving_loss == 0:
            moving_loss = mean_correct
            # print("moving_loss = correct")

        else:
            moving_loss = moving_loss * 0.98 + mean_correct * 0.02


        #output,question,wordgrid = output.detach(),question.detach(),wordgrid.detach()
        output,question = att_vector.detach(),question.detach()

        #cos_sim = cos_similarity(output,question)
        cos_sim = F.mse_loss(prediction,label,reduction="none")

        #Check Attention Weights

        
        yes_answers = cos_sim[label == 1]
        no_answers = cos_sim[label  == 0]

        if yes_answers.size(0) > 0:
            if not answer_dic["yes"]:
                answer_dic["yes"] = yes_answers.tolist()
            else:
                answer_dic["yes"] += yes_answers.tolist()

        if no_answers.size(0) > 0:
            if not answer_dic["no"]:
                answer_dic["no"] = no_answers.tolist()
            else:
                answer_dic["no"] += no_answers.tolist()

        visualize_train(
            global_iteration,run_name,tensorboard_client,
            loss,moving_loss,answer_dic)

        #del loss

        global_iteration += 1

    #valid(epoch + float(i * batch_size / 2325316),tensorboard_client,global_iteration, train_set, model_name=model_name,
    #            load_image=load_image, val_split="train")
    
    return global_iteration,moving_loss
    

def visualize_train(global_iteration,run_name,tensorboard_client,
    loss,moving_loss,answer_dic):
    if (global_iteration % train_progress_iteration != 0):
        return

    # - loss
    loss_dic = {"loss":loss.item()}
    tensorboard_client.append_line(global_iteration,loss_dic,"Training/running_loss")

    # - cosine similarity
    output_dic = {
        "mean_yes":np.mean(answer_dic["yes"]),
        "mean_no":np.mean(answer_dic["no"]),
        "min_yes":np.min(answer_dic["yes"]),
        "max_no":np.max(answer_dic["no"])
        }
    
    acc_dic = {"accuracy":moving_loss.item()}
    tensorboard_client.append_line(global_iteration,acc_dic,"Training/accuracy")
    
    with tensorboard_client.comet_exp.train():
        tensorboard_client.comet_line(loss_dic,"running")
        tensorboard_client.comet_line(acc_dic,"moving")
        tensorboard_client.comet_line(output_dic,"cos_sim")


def valid(epoch,tensorboard_client,global_iteration, valid_set, load_image=True, model_name=None, val_split="val_easy"):
    #run_name = val_split
    
    print("Inside validation ", epoch)
    dataset = tqdm(iter(valid_set))
    model.eval()  # eval_mode
    class_correct = Counter()
    class_total = Counter()
    losses,prediction= [],[]
    
    with torch.no_grad():

        ##for i, (image, question, q_len, answer, answer_class, bboxes, n_bboxes, data_index) in enumerate(tqdm(dataset)):
        for i, (image, question_idx, question, q_len, answer, question_class, embeddings,bboxes,embedding_ids,emb_lengths, data_index) in enumerate(dataset):

            image, question, question_idx, q_len, answer, bboxes, embeddings, emb_lengths, embedding_ids = (
                image.to(device),
                question.to(device),
                question_idx.to(device),
                torch.tensor(q_len).to(device),
                answer.to(device),
                bboxes.to(device),
                embeddings.to(device),
                emb_lengths.to(device),
                embedding_ids.to(device)
            )

            #output,question,wordgrid = vqa_model(image, question, q_len, embeddings, bboxes, emb_lengths)
            output,att_vector,question = model(
                image, question, question_idx, q_len, embeddings, bboxes, embedding_ids, emb_lengths)

            label = answer.clone()
            #label[label == 6.0] = 1.
            #label[label == 7.0] = -1.
            label[label == 6.0] = 1
            label[label == 7.0] = 0

            output = output.squeeze()
            loss = criterion(output,label.float())

            output = output.cpu()
            losses.append(loss.item())

            argmax_output = output > 0.5
            answer = answer.cpu() == 6
            
            correct = argmax_output == answer
            for c, class_ in zip(correct, question_class):
                if c:  # if correct
                    class_correct[class_] += 1
                class_total[class_] += 1

            prediction.append([data_index.numpy(),answer.numpy().astype(int),output.numpy()])

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

    #visualize_val(
    #    global_iteration,tensorboard_client,val_split,len(dataset),epoch,
    #    class_total,class_correct,losses,
    #    prediction)

    tmp_visualize_val(tensorboard_client,val_split,losses,prediction)

    return prediction,total_score

def tmp_visualize_val(tensorboard_client,val_split,losses,prediction):

    total_loss = mean_squared_error(prediction[2],prediction[1])

    mean_yes_values = np.mean(prediction[2][prediction[1] == 1])
    mean_no_values = np.mean(prediction[2][prediction[1] == 0])

    comet_dic = {
        "loss":total_loss,
        "yes_output":mean_yes_values,
        "no_output":mean_no_values
    }

    with tensorboard_client.comet_exp.test():
        tensorboard_client.comet_line(comet_dic,f"{val_split}_mean")


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
    param_path = sys.argv[1]
    params = json.load(open(param_path,"r"))

    global data_path,load_image,reverse_question,lr
    global num_workers,batch_size,weight_decay,device
    global saving_epoch,train_progress_iteration,train_visualization_iteration,validation_epoch

    
    load_image = params["training_params"]["load_image"]
    reverse_question = params["training_params"]["reverse_question"]
    lr = params["training_params"]["lr"]
    weight_decay = params["training_params"]["weight_decay"]
    batch_size = params["training_params"]["batch_size"]
    num_workers = params["training_params"]["n_workers"]
    device = params["training_params"]["device"]
    n_epoch = params["training_params"]["n_epoch"]

    data_path = params["data_dir"]
    load_from_hdf5 = params["load_from_hdf5"]
    model_name = params["model_name"]
    train_file = params["train_file"]
    val_file = params["val_file"]
    exp_name = params["exp_name"]

    saving_epoch = params["visu_params"]["saving_epoch"]
    train_progress_iteration = params["visu_params"]["train_progress_iteration"]
    train_visualization_iteration = params["visu_params"]["train_visualization_iteration"]
    validation_epoch = params["visu_params"]["validation_epoch"]

    heads = params["model_params"]["heads"]
    layers = params["model_params"]["layers"]
    d_model = params["model_params"]["d_model"]
    ff_size = params["model_params"]["ff_size"]
    max_pos = params["model_params"]["max_pos"]
   

    with open(os.path.join(data_path,'dic.pkl'), 'rb') as f:
        dic = pickle.load(f)

    tensorboard_client = TensorBoardVisualize(params,"log/",dic)

    n_words = len(dic['word_dic']) + 1
    n_answers = len(dic['answer_dic'])
    load_from_hdf5 = ast.literal_eval(load_from_hdf5)
    print("load_from_hdf5", load_from_hdf5)
    print(n_answers, "n_answers")

    if model_name == "SANVQA" or "SANVQAbeta":
        model = Encoder(
            n_vocab=n_words,d_model=d_model,n_head=heads,n_layer=layers,
            ff_neurons=ff_size,max_pos=max_pos)#n_answers, n_vocab=n_words, encoded_image_size=(14 if load_from_hdf5 == False else 7))
        #model = nn.DataParallel(model)
        load_image = True
        
        model = model.to(device)

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    #criterion = nn.CosineEmbeddingLoss()
    #criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(model.parameters())
    # scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma) # Decays the learning rate of each parameter group by gamma every step_size epochs.

    train_set = DataLoader(
        DVQA(
            data_path,
            transform=transform,
            reverse_question=reverse_question,
            use_preprocessed=True,
            load_image=load_image,
            load_from_hdf5=load_from_hdf5,
            file_name=train_file

        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=collate_data,
    )

    #Debug
    # valid_set_easy = DataLoader(
    #     DVQA(
    #         data_path,
    #         "val_easy",
    #         transform=None,
    #         reverse_question=reverse_question,
    #         use_preprocessed=True,
    #         load_image=load_image,
    #         load_from_hdf5=load_from_hdf5,
    #         file_name=val_file
    #     ),
    #     batch_size=batch_size,# // 2,
    #     num_workers=n_workers,
    #     collate_fn=collate_data,  ## shuffle=False

    # )
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
    
    
    #tensorboard_client.comet_exp.set_model_graph(str(model))
    start_epoch = 0
    checkpoint_epoch = params["start_epoch"]
    if checkpoint_epoch == "newest":
        checkpoint_name = 'checkpoint/checkpoint_{}_{}.model'.format(exp_name,checkpoint_epoch)
        checkpoint_epoch_name = f'checkpoint/checkpoint_{exp_name}_epoch.txt'

        with open(checkpoint_epoch_name,"r") as f:
            start_epoch = int(f.read())
        checkpoint_epoch = start_epoch

        model.load_state_dict(
            torch.load(checkpoint_name, map_location='cuda'))
    

    global_iteration = len(train_set)*start_epoch
    for epoch in range(start_epoch,n_epoch+start_epoch):
        tensorboard_client.set_epoch_step(epoch,global_iteration)
        # if scheduler.get_lr()[0] < lr_max:
        #     scheduler.step()
        print("epoch=", epoch)

        # TODO: add load model from checkpoint
        checkpoint_name = 'checkpoint/checkpoint_' + exp_name + '_{}.model'.format(str(epoch + 1).zfill(3))
        #if os.path.exists(checkpoint_name):
        #     # model.load_state_dict(torch.load(model.state_dict())
        #     model.load_state_dict(
        #         torch.load(checkpoint_name, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        #     continue

        global_iteration,moving_loss = train(
            epoch,
            tensorboard_client,
            global_iteration,
            train_set,
            train_set.dataset.word_class,
            train_set.dataset.answer_class,
            load_image=load_image, model_name=model_name)

        valid(
            epoch,
            tensorboard_client,
            global_iteration,
            train_set,
            load_image=load_image, model_name=model_name)


        

        
        #Debug        
        tensorboard_client.close()

        if (epoch % saving_epoch == 0):
            checkpoint_name = f'checkpoint/checkpoint_{exp_name}_newest.model'
            checkpoint_epoch_name = f'checkpoint/checkpoint_{exp_name}_epoch.txt'
            #prediction_name = f"predictions/prediction_{sys.argv[3]}_newest.pkl"
            with open(checkpoint_epoch_name,"w") as f:
                    f.write(f"{epoch}")
            with open(checkpoint_name, 'wb') as f:
                torch.save(model.state_dict(), f)

            tensorboard_client.comet_exp.log_model(f"model", checkpoint_name, overwrite=True, metadata={"epoch":epoch})
            #pickle.dump(prediction_name,open(prediction_name,"wb"))
        


