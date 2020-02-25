import cv2
import urllib
import numpy as np
#import cloudpickle
#import multiprocessing
#import math

def calc_bbox_coords(bboxes):
    #x,y,w,h --> x,y,x2,y2
    bboxes[:,2] = bboxes[:,0]+bboxes[:,2]
    bboxes[:,3] = bboxes[:,1]+bboxes[:,3]
    
    bboxes = np.round(bboxes*0.5,0).astype(int)
    return bboxes



def get_labels_and_bboxes(annotation):
    labels,bboxes = list(zip(*[(entry["text"],entry["bbox"]) for entry in annotation["texts"]]))
    #scale from 448x448 to 224x224
    bboxes = calc_bbox_coords(np.array(bboxes))
    
    return labels,bboxes


def enlarge_batch_tensor(tensor):
    n_batch,n_channel,n_row,n_col = tensor.shape 
    
    tensor = tensor.view(-1, 1).repeat(1,2).view(n_batch,n_channel,n_row,n_col*2)
    tensor = tensor.transpose(-1,-2).reshape(-1, 1).repeat(1,2).view(n_batch,n_channel,n_col*2,n_row*2).transpose(-1,-2)
    
    return tensor
