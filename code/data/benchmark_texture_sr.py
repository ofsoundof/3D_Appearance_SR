import os
from data import common
from data import srdata
import glob
import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class BenchmarkTextureSR(srdata.SRData):
    def __init__(self, args, train=True):
        super(BenchmarkTextureSR, self).__init__(args, train, benchmark=True)

    # def _scan(self):
    #     list_hr = []
    #     list_lr = [[] for _ in self.scale]
    #     for entry in os.scandir(self.dir_hr):
    #         filename = os.path.splitext(entry.name)[0]
    #         list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
    #         for si, s in enumerate(self.scale):
    #             list_lr[si].append(os.path.join(self.dir_lr, filename + self.ext))
    #             # list_lr[si].append(os.path.join(
    #             #     self.dir_lr,
    #             #     'X{}/{}x{}{}'.format(s, filename, s, self.ext)
    #             # ))
    #
    #     list_hr.sort()
    #     for l in list_lr:
    #         l.sort()
    #
    #     return list_hr, list_lr

    def _scan(self):
        list_hr = []
        list_lr = []
        for s in self.scale:
            list_hr.append(sorted(glob.glob(self.dir_hr)))
            list_lr.append(sorted(glob.glob(self.dir_lr.format(s))))

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.args.data_test, self.args.data_test_texture_sr)
        self.dir_hr = self.apath + '/x1/Images/*.png'
        self.dir_lr = self.apath + '/x{}/Images/*.png'
        self.ext = '.png'
