import comet_ml

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
import seaborn as sns

from model import Conv2dBatchAct


class TensorBoardVisualize():

    def __init__(self, experiment_name, logdir, dic, hyperparam={"hyper":1}):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = experiment_name+"_"+current_time
        self.tensorboard_writer = SummaryWriter(
            log_dir=pth.join(logdir,experiment_name),
            filename_suffix=experiment_name)

        self.comet_exp = comet_ml.Experiment(project_name="masterthesis")
        self.comet_exp.log_parameters(hyperparam)

        self.comet_exp.log_asset("train_find_text_conv.py")
        self.comet_exp.log_asset("visualize.py")
        self.comet_exp.log_asset("dataset_find_text_conv.py")
        self.comet_exp.log_asset("model_find_text_conv.py")      

        self.word_dic = {v: k for k, v in dic['word_dic'].items()}
        self.answer_dic = {v: k for k, v in dic['answer_dic'].items()}

        self.word_vect = np.vectorize(lambda x: self.word_dic[x] if x > 0 else "")
        self.answer_vect = np.vectorize(lambda x: self.answer_dic[x])

        self.hooks = {}

        self.epoch = 0
        self.step = 0

    def set_epoch_step(self,epoch,step):
        self.epoch = epoch
        self.step = step


    def register_hook(self,key,hook):
        self.hooks[key] = hook

    def append_histogram(self, x, y, chart):

        self.tensorboard_writer.add_histogram(chart, y, x)

        self.comet_exp.log_histogram_3d(y, name=chart, step=self.step)
        #self.tensorboard_writer.close()

    def append_line(self, x, y_dic, chart):
        
        self.tensorboard_writer.add_scalars(chart, y_dic, x)
        #self.tensorboard_writer.close()

    def comet_line(self,y_dic,prefix):
        self.comet_exp.log_metrics(y_dic,prefix=prefix,epoch=self.epoch,step=self.step)

    def comet_image(self,images,chart):

        for i,comet_image in enumerate(images):
            self.comet_exp.log_image(
                comet_image.squeeze(0), name=f"{chart}_{i}", 
                image_format="png",
                image_channels="first", step=self.step)

    def add_images(self,images,chart):

        self.tensorboard_writer.add_images(
                chart, images, global_step=self.step, 
                walltime=None, dataformats='NCHW')

        self.comet_image(images,chart)

    # def add_conv2(self,x,module,chart,hook_name,mask,n_act,suffix=""):

    #     #weights and gradients
    #     weights = module.weight.data.cpu().numpy()
    #     gradients = module.weight.grad.cpu().numpy()
    #     self.append_histogram(x, weights.reshape(-1), f"{chart}_weights")
    #     self.append_histogram(x, gradients.reshape(-1), f"{chart}_gradients")

    #     #need hook
    #     act_hook = self.hooks[hook_name]
    #     act = act_hook.get_features()[mask][:n_act].mean(1,keepdim=True).cpu()
    #     self.add_images(
    #        x,
    #        act,
    #        f"{chart}_activations{suffix}")

    def add_conv2(self,module,chart,hook_name,mask,n_act,suffix=""):

        #weights and gradients
        if isinstance(module,Conv2dBatchAct):
            module = module.conv2d_batch_act[0]
        weights = module.weight.data.cpu().numpy()
        gradients = module.weight.grad.cpu().numpy()

        #self.comet_exp.log_histogram_3d(weights, name=f"{chart}_weights", step=self.step)
        #self.comet_exp.log_histogram_3d(gradients, name=f"{chart}_gradients", step=self.step)
        
        self.append_histogram(self.step, gradients.reshape(-1), f"{chart}_gradients")
        self.append_histogram(self.step, weights.reshape(-1), f"{chart}_weights")
        
        #need hook
        act_hook = self.hooks[hook_name]
        act = act_hook.get_features()[mask][0].unsqueeze(1).cpu()
        act = act - act.min()
        act = act / (act.max() - act.min())
        self.add_images(
           act,
           f"{chart}_act_first_image{suffix}")

    def add_figure_with_question(self,x,image,question,answer,output,index,chart,suffix=""):
        norm_img = mpl.colors.Normalize(vmin=-1,vmax=1)
        visu_question = self.word_vect(question)
        visu_answer = self.answer_vect(answer)
        visu_output = self.answer_vect(output)

        batch_size,channel,h,w = image.shape

        figures = []
        for idx in range(batch_size):
            fig = plt.figure()
            a = fig.add_subplot(111)

            plt.imshow(
                norm_img(np.transpose(image[idx],[1,2,0]).reshape((h,w))),
                vmin=0.,vmax=1.,cmap="gray")
            a.text(0, 0, textwrap.fill(
                    f"{index[idx]}: " + " ".join(visu_question[idx]) + f" Answer/Output: {visu_answer[idx]}/{visu_output[idx]}",
                    60),wrap=True,ha='left',va='bottom')

            figures.append(fig)
            self.comet_exp.log_figure(figure_name=f"{chart}/sample{suffix}_{idx}", figure=fig, overwrite=False, step=self.step)
        
        self.tensorboard_writer.add_figure(
            f"{chart}/sample{suffix}",
            figures,
            x)

        






    def close(self):
        self.tensorboard_writer.close()


class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = [0,0]
    def hook_fn(self, module, input, output):
        self.features[output.device.index] = output.detach() #torch.tensor(output,requires_grad=True).cuda()
    def get_features(self):
        if self.features[1] == 0:
            return self.features[0]
        return torch.cat(self.features,0)
    def close(self):
        self.hook.remove()