__author__ = 'yawli'

import os

from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data
import random

class SRData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0
        self.color = 'RGB' if args.n_colors == 3 else 'Y'
        self.model_one = args.model_one == 'one'
        self.model_flag = args.model
        self.data_train = args.data_train
        self.subset = args.subset
        self.normal_lr = args.normal_lr == 'lr'
        self.input_res = args.input_res
        self._set_filesystem(args.dir_data)

        def _load_bin():
            self.images_hr = np.load(self._name_hrbin())
            self.images_lr = [
                np.load(self._name_lrbin(s)) for s in self.scale
            ]

        if args.ext == 'img' or benchmark:
            self.images_hr, self.images_lr, self.normals_lr, self.masks_lr = self._scan()
        elif args.ext.find('sep') >= 0:
            self.images_hr, self.images_lr, self.normals_lr, self.masks_lr = self._scan()
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_hr:

                    hr = misc.imread(v)
                    name_sep = v.replace(self.ext, '.npy')
                    np.save(name_sep, hr)
                    # from IPython import embed; embed(); exit()
                for si, s in enumerate(self.scale):
                    for n in range(len(self.images_lr[si])):
                        v = self.images_lr[si][n]
                        #from IPython import embed; embed(); exit()
                        lr = misc.imread(v)
                        name_sep = v.replace(self.ext, '.npy')
                        np.save(name_sep, lr)

                        v = self.normals_lr[si][n]
                        lr = misc.imread(v)
                        name_sep = v.replace(self.ext, '.npy')
                        np.save(name_sep, lr)

                        v = self.masks_lr[si][n]
                        lr = np.expand_dims(misc.imread(v), 2)
                        name_sep = v.replace(self.ext, '.npy')
                        np.save(name_sep, lr)
                        # from IPython import embed; embed(); exit()
            self.images_hr = [
                v.replace(self.ext, '.npy') for v in self.images_hr
            ]
            self.images_lr = [
                [v.replace(self.ext, '.npy') for v in self.images_lr[i]]
                for i in range(len(self.scale))
            ]
            self.normals_lr = [
                [v.replace(self.ext, '.npy') for v in self.normals_lr[i]]
                for i in range(len(self.scale))
            ]
            self.masks_lr = [
                [v.replace(self.ext, '.npy') for v in self.masks_lr[i]]
                for i in range(len(self.scale))
            ]
            # from IPython import embed; embed(); exit()
        elif args.ext.find('bin') >= 0:
            try:
                if args.ext.find('reset') >= 0:
                    raise IOError
                print('Loading a binary file')
                _load_bin()
            except:
                print('Preparing a binary file')
                bin_path = os.path.join(self.apath, 'bin')
                if not os.path.isdir(bin_path):
                    os.mkdir(bin_path)

                list_hr, list_lr = self._scan()
                hr = [misc.imread(f) for f in list_hr]
                np.save(self._name_hrbin(), hr)
                del hr
                for si, s in enumerate(self.scale):
                    lr_scale = [misc.imread(f) for f in list_lr[si]]
                    np.save(self._name_lrbin(s), lr_scale)
                    del lr_scale
                _load_bin()
        else:
            print('Please define data type')
    # from IPython import embed; embed(); exit()
    def _scan(self):
        raise NotImplementedError

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_hrbin(self):
        raise NotImplementedError

    def _name_lrbin(self, scale):
        raise NotImplementedError

    def __getitem__(self, idx):
        lr, nl, mk, hr, filename = self._load_file(idx)
        lr, nl, mk, hr = self._get_patch(lr, nl, mk, hr)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        #print('The size of lr, hr images are {}, {}'.format(lr.shape, hr.shape))
        lr_tensor, nl_tensor, mk_tensor, hr_tensor = common.np2Tensor([lr, nl, mk, hr], self.args.rgb_range)
        # if self.model_flag.lower() == 'finetune':
        return lr_tensor, nl_tensor, mk_tensor, hr_tensor, filename
        # else:
        #     return lr_tensor, hr_tensor, filename

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        # from IPython import embed; embed()
        # print('The hr images are {}'.format(self.images_hr))
        lr = self.images_lr[self.idx_scale][idx]
        nl = self.normals_lr[self.idx_scale][idx]
        mk = self.masks_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]

        # print(self.images_hr)
        # print('......................................................')
        # print(hr)
        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            lr = misc.imread(lr)
            nl = misc.imread(nl)
            mk = misc.imread(mk)
            hr = misc.imread(hr)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            lr = np.load(lr)
            nl = np.load(nl)
            mk = np.load(mk)
            hr = np.load(hr)

        else:
            filename = str(idx + 1)
        #print('The resolution of lr, hr images are {}, {}'.format(lr.shape, hr.shape))
        filename = os.path.splitext(os.path.split(filename)[-1])[0]
        #print('the filename is {}'.format(filename))
        return lr, nl, mk, hr, filename

    def _get_patch(self, lr, nl, mk, hr):
        patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            #from IPython import embed; embed(); exit()
            lr, nl, mk, hr = self.get_patch(
                lr, nl, mk, hr, patch_size, scale, multi_scale=multi_scale
            )
            lr, nl, mk, hr = common.augment([lr, nl, mk, hr])
            lr = common.add_noise(lr, self.args.noise)
        else:
            ih, iw = lr.shape[0:2]
            hr = hr[0:ih * scale, 0:iw * scale]
            if not self.normal_lr:
                nl = nl[0:ih * scale, 0:iw * scale]

        return lr, nl, mk, hr

    def get_patch(self, img_in, nml_in, msk_in, img_tar, patch_size, scale, multi_scale=False):
        ih, iw = img_in.shape[:2]

        p = scale if multi_scale else 1
        tp = p * patch_size
        ip = tp // scale

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = scale * ix, scale * iy
        if self.input_res == 'lr':
            img_in = img_in[iy:iy + ip, ix:ix + ip, :]
            msk_in = msk_in[iy:iy + ip, ix:ix + ip]
            img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]
            if self.normal_lr:
                nml_in = nml_in[iy:iy + ip, ix:ix + ip, :]
            else:
                nml_in = nml_in[ty:ty + tp, tx:tx + tp, :]
        else:
            img_in = img_in[iy:iy + ip, ix:ix + ip, :]
            msk_in = msk_in[iy:iy + ip, ix:ix + ip]
            img_tar = img_tar[iy:iy + ip, ix:ix + ip, :]
            nml_in = nml_in[iy:iy + ip, ix:ix + ip, :]

        return img_in, nml_in, msk_in, img_tar

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

