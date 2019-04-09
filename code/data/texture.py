__author__ = 'Yawei Li'

import os

from data import common
from data import srtexture

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data
import glob

class texture(srtexture.SRData):
    def __init__(self, args, train=True):
        super(texture, self).__init__(args, train)

        if self.subset == '.':
            self.num_all = 24
            self.num_split = 12
            if train:
                self.repeat = args.test_every // (self.num_split // args.batch_size)
        else:
            self.num_all = self.all[self.subset_idx[0]]
            if (self.train and self.model_one) or (not self.train and not self.model_one):
                self.num_split = self.split[self.subset_idx[0]]
            else:
                self.num_split = self.num_all - self.split[self.subset_idx[0]]

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        list_lr_normal = [[] for _ in self.scale]
        list_lr_mask = [[] for _ in self.scale]
        self.subset_idx = list(range(4)) if self.subset == '.' else [self.set.index(self.subset)]
        for s in self.subset_idx:
            dir_hr = os.path.join(self.apath, self.set[s], 'x1/Texture/*.png')
            list_hr_set = sorted(glob.glob(dir_hr))
            if self.data_train == 'texture':
                if (self.train and self.model_one) or (not self.train and not self.model_one):
                    list_hr_split = list_hr_set[:self.split[s]]
                    list_hr += list_hr_split
                else:
                    list_hr_split = list_hr_set[self.split[s]:]
                    list_hr += list_hr_split
            else:
                list_hr += list_hr_set
        for si, s in enumerate(self.scale):
            list_lr[si] = [n.replace('x1', 'x{}'.format(s)) for n in list_hr]
            if self.normal_lr:
                # from IPython import embed; embed(); exit()
                list_lr_normal[si] = [n.replace('x1', 'x{}'.format(s)).replace('Texture/', 'normal/').replace('Texture.png', 'normal.png') for n in list_hr]
            else:
                list_lr_normal[si] = [n.replace('Texture/', 'normal/').replace('Texture.png', 'normal.png') for n in list_hr]
            list_lr_mask[si] = [n.replace('x1', 'x{}'.format(s)).replace('Texture/', 'mask/').replace('Texture.png', 'mask.png') for n in list_hr]
        # from IPython import embed; embed(); exit()
        return list_hr, list_lr, list_lr_normal, list_lr_mask

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + 'Texture_map'
        self.set = ['MiddleBury', 'ETH3D', 'Collection', 'SyB3R']
        self.ext = '.png'
        self.split = [1, 6, 3, 2] #[1, 7, 3, 1]
        self.all = [2, 13, 6, 3]
        # if self.model_one:
        #     self.split = [1, 6, 3, 2]

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        )

    def __len__(self):
        if self.train:
            if self.subset == '.':
                return self.num_split * self.repeat
            else:
                return 2400
        else:
            if self.data_train == 'texture':
                return self.num_split
            else:
                return self.num_all

    def _get_index(self, idx):
        if self.train:
            return idx % self.num_split
        else:
            return idx


