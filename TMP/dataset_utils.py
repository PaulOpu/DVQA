import cv2
import urllib
import numpy as np
import cloudpickle
import multiprocessing
import math


def load_image_from_url(path):
    with urllib.request.urlopen(path) as url:
        arr = np.asarray(bytearray(url.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
    return img


def resize_boxes(boxes,img_shape,scale_shape):
    #retun actual dimensions of the bbox for image features
    #scale_shape = (image_w,image_h)
    #img_shape = img.shape
    width_ratio = img_shape[1]/scale_shape[0]
    height_ratio = img_shape[0]/scale_shape[1]
    
    boxes[:, :4] = boxes[:, :4] * np.array(
        [width_ratio,
         height_ratio,
         width_ratio,
         height_ratio]
    ).reshape((1,4))
    
    return boxes


# +
def save_dic(obj, name ):
    with open(name, 'wb') as f:
        cloudpickle.dump(obj, f)

def load_dic(name ):
    with open(name, 'rb') as f:
        return cloudpickle.load(f)


# +
def split_work_in_threads(array,func,kernels,splitting=True,concat_func=None,**kwargs):
    #Function that shared the work among the usable kernels
    #param df - dataframe ("text","category")
    #param func - preprocessing step
    #param kernels - number of usable kernels
    #return concatinated array from all kernels
    
    if splitting:
        #splitting dataframe
        array_pieces = np.array_split(array, kernels)
    else:
        array_pieces = array
    
    processes = []
    queue = multiprocessing.Queue()
    
    #each kernel gets a process assigned with a piece of the df
    for i in range(kernels):
        p = multiprocessing.Process(
                target=assign_work_to_thread,
                args=(array_pieces[i],func,queue,i,kwargs))
        processes.append(p)
        p.start()
    
    #collect the results
    results = []
    for i in range(kernels):
        results.append(queue.get())
    
    if concat_func is not None:
        return concat(results)
    else:
        return results

def assign_work_to_thread(df,func,results,i,kwargs):
    #Helper Function to deal with work splitting
    #param df - piece of dataframe
    #param func - prepr. step
    #param results - thread queue
    #param i - index of thread (only internally used)
    #return results.put(df) - add cleaned df to result list
    
    df = func(df,**kwargs)
    results.put(df)
# -


