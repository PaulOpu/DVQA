import os.path as pth
import json
import numpy as np
import time
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardVisualize():

    def __init__(self, experiment_name, logdir):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_writer = SummaryWriter(
            log_dir=pth.join(logdir,experiment_name+"_"+current_time),
            filename_suffix=experiment_name)

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


    def close(self):
        self.tensorboard_writer.close()


class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = output.detach() #torch.tensor(output,requires_grad=True).cuda()
    def close(self):
        self.hook.remove()