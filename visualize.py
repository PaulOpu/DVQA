import os.path as pth
import json
import numpy as np
import time
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib as mpl
import textwrap


class TensorBoardVisualize():

    def __init__(self, experiment_name, logdir, dic):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_writer = SummaryWriter(
            log_dir=pth.join(logdir,experiment_name+"_"+current_time),
            filename_suffix=experiment_name)

        self.word_dic = {v: k for k, v in dic['word_dic'].items()}
        self.answer_dic = {v: k for k, v in dic['answer_dic'].items()}

        self.word_vect = np.vectorize(lambda x: self.word_dic[x] if x > 0 else "")
        self.answer_vect = np.vectorize(lambda x: self.answer_dic[x])

        self.hooks = {}

    def get_current_epoch(iteration):

        return iteration / self.batch_size


    def register_hook(self,key,hook):
        self.hooks[key] = hook

    def append_histogram(self, x, y, chart):
        self.tensorboard_writer.add_histogram(chart, y, x)
        #self.tensorboard_writer.close()

    def append_line(self, x, y_dic, chart):

        self.tensorboard_writer.add_scalars(chart, y_dic, x)
        #self.tensorboard_writer.close()

    def add_images(self,x,images,chart):
        self.tensorboard_writer.add_images(
                chart, images, global_step=x, 
                walltime=None, dataformats='NCHW')

    def add_conv2(self,x,module,chart,hook_name,mask,n_act,suffix=""):

        #weights and gradients
        weights = module.weight.data.cpu().numpy()
        gradients = module.weight.grad.cpu().numpy()
        self.append_histogram(x, weights.reshape(-1), f"{chart}/weights")
        self.append_histogram(x, gradients.reshape(-1), f"{chart}/gradients")

        #need hook
        act_hook = self.hooks[hook_name]
        act = act_hook.features[mask][:n_act].sum(1,keepdim=True).cpu()
        self.add_images(
           x,
           act,
           f"{chart}/activations{suffix}")

    def add_figure_with_question(self,x,image,question,answer,output,chart,suffix=""):
        norm_img = mpl.colors.Normalize(vmin=-1,vmax=1)
        visu_question = self.word_vect(question)
        visu_answer = self.answer_vect(answer)
        visu_output = self.answer_vect(output)

        figures = []
        for idx in range(image.shape[0]):
            fig = plt.figure()
            a = fig.add_subplot(111)

            plt.imshow(
                norm_img(np.transpose(image[idx],[1,2,0])),
                vmin=0.,vmax=1.)
            a.text(0, 0, textwrap.fill(
                    " ".join(visu_question[idx]) + f"Answer/Output: {visu_answer[idx]}/{visu_output[idx]}",
                    60),wrap=True,ha='left',va='bottom')

            figures.append(fig)
        
        self.tensorboard_writer.add_figure(
            f"{chart}/sample{suffix}",
            figures,
            x)






    def close(self):
        self.tensorboard_writer.close()


class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.detach() #torch.tensor(output,requires_grad=True).cuda()
    def close(self):
        self.hook.remove()