import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data
import glob
class DIV2KSUB(srdata.SRData):
    def __init__(self, args, train=True):
        super(DIV2KSUB, self).__init__(args, train)
        self.repeat = max(round(args.test_every / (args.n_train / args.batch_size)), 1)
        self.n_train = args.n_train
    def _scan(self):
        list_hr = sorted(glob.glob(os.path.join(self.dir_hr, '*.png')))
        list_lr = [sorted(glob.glob(os.path.join(self.dir_lr + '{}'.format(s), '*.png'))) for s in self.scale]
        #for si, s in enumerate(self.scale):

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + '/DIV2K'
        self.dir_hr = os.path.join(self.apath, 'GT_sub')
        self.dir_lr = os.path.join(self.apath, 'GT_sub_bicLRx')
        self.ext = '.png'

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
            return self.n_train * self.repeat #len(self.images_hr) * self.repeat
        else:
            return self.n_train #len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % self.n_train #len(self.images_hr)
        else:
            return idx


# # block2
# class VarBlockSimple(nn.Module):
#     '''
#     regression block used for CARN
#     '''
#
#     def __init__(self, conv=common.default_conv, n_feats=64, kernel_size=3, reg_act=nn.Softplus(), rescale=1):
#         super(VarBlockSimple, self).__init__()
#         conv_mask = [conv(n_feats, 1, kernel_size=5), nn.PReLU(), conv(1, 1, kernel_size=5), reg_act]
#         conv_body = [conv(n_feats, n_feats, kernel_size), nn.PReLU()]
#         self.conv_mask = nn.Sequential(*conv_mask)
#         self.conv_body = nn.Sequential(*conv_body)
#
#     def forward(self, x):
#         #x = torch.matmul(x, self.conv_mask(x))
#         res = self.conv_body(self.conv_mask(x) * x)
#         x = res + x
#         return x
# #block3
# class VarBlockSimple(nn.Module):
#     '''
#     regression block used for CARN
#     '''
#
#     def __init__(self, conv=common.default_conv, n_feats=64, kernel_size=3, reg_act=nn.Softplus(), rescale=1):
#         super(VarBlockSimple, self).__init__()
#         conv_mask = [conv(n_feats, 1, kernel_size=5), nn.PReLU(), conv(1, 1, kernel_size=5), reg_act]
#         conv_body = [conv(n_feats, n_feats, kernel_size), nn.PReLU()]
#         conv_tail = [conv(n_feats, n_feats, kernel_size), nn.PReLU()]
#         self.conv_mask = nn.Sequential(*conv_mask)
#         self.conv_body = nn.Sequential(*conv_body)
#         self.conv_tail = nn.Sequential(*conv_tail)
#
#     def forward(self, x):
#         #x = torch.matmul(x, self.conv_mask(x))
#         res = self.conv_body(self.conv_mask(x) * x)
#         x = res + self.conv_tail(x)
#         return x
