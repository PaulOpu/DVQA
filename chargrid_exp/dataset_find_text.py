# -*- coding: utf-8 -*-
import os
import pickle

import h5py
import numpy as np
import time
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


import sys
import json
sys.path.append('/workspace/st_vqa_entitygrid/solution/')
#sys.path.append('/project/paul_op_masterthesis/st_vqa_entitygrid/solution/')
from dvqa import get_labels_and_bboxes,generate_wordgrid
from figureqa import load_vectorizer


resize = transforms.Resize([128, 128])

transform = transforms.Compose([
    transforms.Pad(8),
    transforms.RandomCrop([128, 128]),
    transforms.RandomRotation(2.8),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# Might be something we want to look at ...
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

YES_transform = transforms.ToTensor()

IMG_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# TODO: resnet152 the network has been trained with color images of size 224x224.
# If your images are for instance grayscale you have to copy the single channel three times.
# If your images are of a different size, you can resize them or get a 224x224 crop out of them.
# Keep in mind that you can get your features faster by creating a big tensor with
# let’s say 100 images (shape 100x3x224x224) and process them in a single round.

# Therefore as long as the input image size makes the AvgPool output tensors of size 1x2048x1x1, there is no problem.
# But if the input size is not 224x224, it is cropped by ResNet implicitly at AvgPool layer.
# Note: we don't use AvgPool, so any size works...

category = {'0': 'count',
            '1': 'count',
            '2': 'count',
            '3': 'count',
            '4': 'count',
            '5': 'count',
            '6': 'count',
            '7': 'count',
            '8': 'count',
            '9': 'count',
            '10': 'count',
            'blue': 'color',
            'brown': 'color',
            'cyan': 'color',
            'yellow': 'color',
            'gray': 'color',
            'green': 'color',
            'purple': 'color',
            'red': 'color',
            'rubber': 'material',
            'metal': 'material',
            'large': 'size',
            'small': 'size',
            'cylinder': 'shape',
            'cube': 'shape',
            'sphere': 'shape',
            'no': 'exist',
            'yes': 'exist'}

class DVQA(Dataset):
    def __init__(self, root, split='train', transform=None,
                 reverse_question=False, use_preprocessed=False, load_image=True,
                 hdf5_image_dir="data/", load_from_hdf5=True, file_name="train.pkl"):
        #Debug
        with open(os.path.join(root,file_name), 'rb') as f:
            self.data = pickle.load(f)

        with open(os.path.join(root,'dic.pkl'), 'rb') as f:
            self.dic = pickle.load(f)
        #SANDY: Add placeholder for dynamic encoding (index 0-29 reserved for DEM)
        self.answer_class = {v: k for k, v in self.dic['answer_dic'].items()}  # answer i2a
        self.word_class = {v: k for k, v in self.dic['word_dic'].items()}  # word i2w
        self.OOV_index = 0
        self.max_word_idx = max(self.word_class.keys())  # largest vocab word index in train
        self.transform = transform
        self.root = root
        self.split = split
        self.reverse_question = reverse_question
        self.use_preprocessed = use_preprocessed
        self.load_image = load_image
        print("load_image", self.load_image)
        self.load_from_hdf5 = load_from_hdf5

        if self.load_image and self.load_from_hdf5:
            print("Loading from hdf5")
            self.h = h5py.File(os.path.join(root, self.split + '_IMAGES_.hdf5'), 'r')
            self.imgs = self.h['images']
        else:
            print("either not self.load_image or not self.load_from_hdf5")

        #Chargrid: load image_id2metadata.json
        self.id2metadata = json.load(open(os.path.join(root, f"{split}_id2metadata.json"),"r"))

        #Chargrid: load vectorizer
        #self.vectorizer = load_vectorizer(os.path.join(root,"dvqa_bag_of_characters.pkl"))
        
        ##self.bb = h5py.File(os.path.join(root, self.split + '_bboxes.hdf5'), 'r')
        ##self.bboxes = self.bb['emb_bboxes']
        ##self.bbox_len = self.bb['emb_bboxes_len']

        self.c = h5py.File(os.path.join(root, self.split + '_chargrid.h5py'), 'r')
        self.chargrids = self.c['chargrid']

        #Chargrid: Question2Character
        self.label_encoder = pickle.load(open(os.path.join(root,"label_encoder.pkl"),"rb"))




    def __getitem__(self, index):
        hdf5_idx_for_this_image, imgfile, question, answer, question_type = self.data[
            index]  # question['image'], question_token, answer, question['template_id'],  # question/answer class

        # TODO: check and alert non-exist data items https://discuss.pytorch.org/t/possible-to-skip-bad-items-in-data-loader/3439
        # Good practices tho: If the data is not growing during training phase, I would treat this as a data cleanup task before training.
        # if not os.path.exists(os.path.join(self.root, 'images',
        #                                   self.split, imgfile)):
        #     return

        if self.load_image:
            # if self.use_preprocessed is False:
            #     img = Image.open(os.path.join(self.root, 'images',
            #                                   "unsplitted_images", imgfile)).convert('RGB')
            #     img = resize(img)
            #
            # else:
            #     img = Image.open(os.path.join(self.root, 'images',
            #                                   "unsplitted_images",
            #                                   imgfile)).convert('RGB')
            # start = time.time()
            if self.load_from_hdf5:

                img = self.imgs[hdf5_idx_for_this_image]  # (3,224,224)

                # print("finished load_from_hdf5", time.time() - start)
                # start=time.time()

                # TODO: earlier mistake, should not transpose in preprocess.py
                img = img.transpose(1, 2, 0)

                # print("finished transpose", time.time() - start)
                # start = time.time()

                img = IMG_transform(img)

                # print("finished IMG_transform", time.time() - start)


            else:
                img = Image.open(os.path.join(self.root, 'images',
                                              "unsplitted_images", imgfile)).convert('RGB')
                # img=resize(img)
                img = IMG_transform(img)
                # input(img.shape)

        else:
            img = torch.Tensor(np.zeros(1))

        # answer_class = category[self.answer_class[answer]]
        question_class = question_type

        # TODO: transform based on DVQA paper
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # else:
        #     img = eval_transform(img)

        # TODO!: switch question tensor (of val_easy or val_hard) position whose value > self.max_word_idx to self.OOV_index
        
        #---------
        #Use Character Encoding of Question
        #question = self.label_encoder.transform(
        #    list(" ".join([self.word_class[i].lower() for i in question])))
        #---------
        question = np.array(question)
        question[question > self.max_word_idx] = self.OOV_index

        if self.reverse_question:  # TODO: test this variant for QUES model
            question = question[::-1]

        # #Chargrid: get labels/bboxes and vectorize
        # metadata = self.id2metadata[imgfile]
        # labels,bboxes = get_labels_and_bboxes(metadata)
        # torch_bboxes = torch.tensor(bboxes)
        
        # emb_labels = self.vectorizer.transform(labels).toarray()
        # #normalized chargrid
        # label_sum = np.sum(emb_labels,1)
        # torch_labels = torch.tensor(emb_labels / label_sum[:, np.newaxis])
        
        # n_label,n_dim = emb_labels.shape

        ##n_bbox = torch.tensor(self.bbox_len[index])
        ##torch_bboxes = torch.tensor(self.bboxes[index,:n_bbox])

        #chargrid = torch.tensor(self.chargrids[hdf5_idx_for_this_image])
        #chargrid = torch.zeros((224,224))
        chargrid = torch.tensor(generate_wordgrid(self.id2metadata[imgfile],self.dic))

        ##return img, question, len(question), answer, question_class, torch_bboxes, n_bbox, index  # answer_class
        return img, question, len(question), answer, question_class, chargrid, index

    def __len__(self):
        return len(self.data)


def collate_data(batch):
    

    images, lengths, answers, question_class = [], [], [], []
    ##bbox_batch,bbox_len_batch = [],[]
    
    batch_size = len(batch)
    chargrid_batch = torch.zeros((batch_size,224,224),dtype=torch.long)
    # max_labels = max([entry[7] for entry in batch])
    # max_label_dim = batch[0][5].shape[-1]

    # batch_labels = torch.zeros((batch_size,max_labels,max_label_dim))
    # batch_bboxes = np.zeros((batch_size,max_labels,4),int)#torch.zeros((batch_size,max_labels,4))
    # batch_n_labels = torch.zeros((batch_size),dtype=torch.int32)



    img_ids = torch.zeros((batch_size))

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True) 

    for i, b in enumerate(sort_by_len):
        #Chargrid: collate labels/bboxes 
        ##image, question, length, answer, class_, bboxes, n_bbox, index = b  # destructure a batch's data
        image, question, length, answer, class_, chargrid, index = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)
        question_class.append(class_)

        ##bbox_batch.append(bboxes)
        ##bbox_len_batch.append(torch.tensor([i]).repeat(n_bbox))
        chargrid_batch[i] = chargrid

        # bboxes = np.clip(bboxes,a_min=0,a_max=224)

        # batch_labels[i,:n_label,:] = labels
        # batch_bboxes[i,:n_label,:] = bboxes
        # batch_n_labels[i] = n_label
        img_ids[i] = index

    return torch.stack(images), torch.from_numpy(questions), \
           lengths, torch.LongTensor(answers), question_class, \
           chargrid_batch, img_ids
           ##torch.cat(bbox_batch).long(),torch.cat(bbox_len_batch),img_ids 
           #batch_labels, batch_bboxes, batch_n_labels, img_ids 
           
