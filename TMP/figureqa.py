import cv2
import urllib
import numpy as np
import cloudpickle
import multiprocessing
import math
import random
import PIL.ImageOps as ImageOps
import itertools
import pickle
from sklearn.feature_extraction.text import CountVectorizer


def char_split(document):
    return list(document.lower())


def vectorize_characters(documents):
    #each document is one label in the chart and will be split by each character

    #all documents in one list
    corpus = list(itertools.chain(*documents))

    #vectorize
    vec_counter = CountVectorizer(tokenizer=char_split)
    vec_counter.fit(corpus)
    
    return vec_counter

def load_vectorizer(path):
    return pickle.load(open(path,"rb"))

def get_bbox_dimensions(bbox,factor):
    x = int(bbox["x"]*factor)
    y = int(bbox["y"]*factor)
    w = int(bbox["w"]*factor)
    h = int(bbox["h"]*factor)
    
    return x,y,w,h


# +
def get_padding(input_shape,output_shape):
    
    x_pad,y_pad = np.array(output_shape) - np.array(input_shape)
    
    output_paddings = []
    
    for padding in [x_pad,y_pad]:
        
        side_padding = get_side_pads(padding / 2)
        output_paddings.append(side_padding)
        
    return output_paddings

def get_side_pads(pad_size):
    if pad_size < 0:
        return (0,0)
    if pad_size == int(pad_size):
        pad_size = int(pad_size)
        left_pad,right_pad = pad_size,pad_size
    else:
        left_pad = math.ceil(pad_size)
        right_pad = math.floor(pad_size)
        
    return (left_pad,right_pad)


# +
def get_offset_and_scale(final_dim,annotation,x_rand_pad,y_rand_pad,padding):
    
    image_dimensions = annotation['general_figure_info']["figure_info"]["bbox"]["bbox"]
    old_image_h = image_dimensions["h"]
    old_image_w = image_dimensions["w"]
    max_dim = max([old_image_h,old_image_w])
    scale = final_dim / max_dim
    
    new_image_h = int(old_image_h*scale)
    new_image_w = int(old_image_w*scale)
    
    #padding
    pad_h,pad_w = get_padding([new_image_h,new_image_w],(final_dim,final_dim))
    
    #padding offset
    x_offset = pad_w[0] - x_rand_pad + padding#x_rand_pad - padding
    y_offset = pad_h[0] - y_rand_pad + padding#y_rand_pad - padding
    #x = x + pad_w[0] - x_rand_pad + padding
    
    return x_offset,y_offset,scale

def get_coords(x,y,x2,y2,img_shape):
    #let the bbox stay in the imag dimensions
    x,y = max([0,x]),max([0,y])
    x2,y2 = min([x2,img_shape[1]]),min([y2,img_shape[0]])
    w,h = x2-x,y2-y
        
    return x,y,x2,y2,h,w
    
def check_placement(x,y,x2,y2,img_shape):
    result = False
    if (x > img_shape[1]) or (y > img_shape[0]):
        pass
    elif (x2 < 0) or (y2 < 0):
        pass
    else:
        result = True
    return result

def create_bbox_canvas(vectorizer,annotation,x_rand_pad=0,y_rand_pad=0,padding=0):
    final_dim = 256
    n_channels = len(vectorizer.vocabulary_)
    #final dimension 256 x 256
    
    x_offset,y_offset,scale = get_offset_and_scale(
        final_dim,
        annotation,
        x_rand_pad,
        y_rand_pad,
        padding
        )
    
    img_canvas = np.zeros(
        (
            final_dim,#new_image_h + sum(pad_h),
            final_dim,#new_image_w + sum(pad_w),
            n_channels
        ),
        int
    )
    
    
    #get bounding boxes and labels
    labels_bboxes = extract_labels_bboxes(annotation)
    labels,bboxes = list(zip(*labels_bboxes))
    
    #emb_labels 
    emb_labels = vectorizer.transform(labels).toarray()
    
    
    for idx,bbox in enumerate(bboxes):
        x,y,w,h = get_bbox_dimensions(bbox,scale)
        x,y = x + x_offset, y + y_offset
        x2,y2 = x + w,y + h
        emb_label = emb_labels[idx]
        img_shape = img_canvas.shape
        if check_placement(x,y,x2,y2,img_shape):
            x,y,x2,y2,h,w = get_coords(x,y,x2,y2,img_shape)
            chargrid = np.tile(emb_label,(h,w)).reshape(h,w,n_channels)
            img_canvas[y:y2,x:x2,:] = chargrid
        
        
        
    return img_canvas


# +
def randomcrop_img_chargrid(img,vectorizer,annotation,padding=0):
    img = ImageOps.expand(img, border=padding, fill=0)
    rand_x = random.randint(1,padding*2)
    rand_y = random.randint(1,padding*2)
    img = img.crop(box=[rand_x,rand_y,256+rand_x,256+rand_y])
    
    chargrid = create_bbox_canvas(vectorizer,annotation,x_rand_pad=rand_x,y_rand_pad=rand_y,padding=padding)
    
    return img,chargrid

def randomcrop_img(img,padding=0):
    img = ImageOps.expand(img, border=padding, fill=0)
    rand_x = random.randint(1,padding*2)
    rand_y = random.randint(1,padding*2)
    img = img.crop(box=[rand_x,rand_y,256+rand_x,256+rand_y])
    
    return img,rand_x,rand_y


# -

def get_chargrid_bboxes(annotation,vectorizer,x_rand_pad,y_rand_pad,padding):
    #final dimension 256 x 256
    final_dim = 256
    n_channels = len(vectorizer.vocabulary_)
    
    
    x_offset,y_offset,scale = get_offset_and_scale(
        final_dim,
        annotation,
        x_rand_pad,
        y_rand_pad,
        padding
        )
    
    #get bounding boxes and labels
    labels_bboxes = extract_labels_bboxes(annotation)
    labels,bboxes = list(zip(*labels_bboxes))
    
    #emb_labels 
    emb_labels = vectorizer.transform(labels).toarray()
    replaced_bboxes = np.zeros((len(bboxes),4),int)
    
    placed_bboxes = []
    for idx,bbox in enumerate(bboxes):
        x,y,w,h = get_bbox_dimensions(bbox,scale)
        x,y = x + x_offset, y + y_offset
        x2,y2 = x + w,y + h
        emb_label = emb_labels[idx]
        img_shape = (final_dim,final_dim)
        if check_placement(x,y,x2,y2,img_shape):
            x,y,x2,y2,h,w = get_coords(x,y,x2,y2,img_shape)
            #chargrid = np.tile(emb_label,(h,w)).reshape(h,w,n_channels)
            #img_canvas[y:y2,x:x2,:] = chargrid
            replaced_bboxes[idx,:] = [x,y,x2,y2]
            
            placed_bboxes += [idx]
        
    emb_labels = emb_labels[placed_bboxes,:]
    replaced_bboxes = replaced_bboxes[placed_bboxes,:]
        
    return emb_labels,replaced_bboxes

# +
PIE_CHART_TYPE = "pie"
BAR_CHART_TYPES = ['vbar_categorical','hbar_categorical']
LINE_CHART_TYPES = ['line','dot_line']

def extract_labels_bboxes(annotation):
    chart_type = annotation["type"]
    LEGEND_TYPE = "legend"
    AXIS_TYPE = "axis"
    AXIS_LABELS_TYPE = "axis_labels"
    
    areas = []
    if chart_type == PIE_CHART_TYPE:
        areas.append(LEGEND_TYPE)
    elif chart_type in LINE_CHART_TYPES:
        areas += [LEGEND_TYPE,AXIS_TYPE,AXIS_LABELS_TYPE]
    elif chart_type in BAR_CHART_TYPES:
        areas += [AXIS_TYPE,AXIS_LABELS_TYPE]
        
    #return areas
    
    #create list of labels and bboxes
    labels_bboxes = []
    figure_info = annotation["general_figure_info"]
    
    #title extraction
    #text
    text = figure_info["title"]["text"]
    #bbox y x w h
    bbox = figure_info["title"]["bbox"]
    
    labels_bboxes.append((text,bbox))
    
    #legend extraction
    if LEGEND_TYPE in areas:
        for item in figure_info["legend"]["items"]:
            text = item["label"]["text"]
            bbox = item["label"]["bbox"]
            
            labels_bboxes.append((text,bbox))
    
    #axis extraction
    if AXIS_TYPE in areas:
        x_text = figure_info["x_axis"]["label"]["text"]
        x_bbox = figure_info["x_axis"]["label"]["bbox"]

        y_text = figure_info["y_axis"]["label"]["text"]
        y_bbox = figure_info["y_axis"]["label"]["bbox"]
        
        labels_bboxes.append((x_text,x_bbox))
        labels_bboxes.append((y_text,y_bbox))
        
        
    #axis_labels extraction
    if AXIS_LABELS_TYPE in areas:
        x_texts = figure_info["x_axis"]["major_labels"]["values"]
        x_bboxes =  figure_info["x_axis"]["major_labels"]["bboxes"]

        y_texts = figure_info["y_axis"]["major_labels"]["values"]
        y_bboxes =  figure_info["y_axis"]["major_labels"]["bboxes"]
        
        labels_bboxes += list(zip(*[x_texts,x_bboxes]))
        labels_bboxes += list(zip(*[y_texts,y_bboxes]))
    
    return labels_bboxes
