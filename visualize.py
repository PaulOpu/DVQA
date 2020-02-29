import os.path as pth
import json
import numpy as np
import visdom
import time
import datetime

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

    def close(self):
        self.tensorboard_writer.close()