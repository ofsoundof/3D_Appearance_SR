__author__ = 'yawli'

import numpy as np
import os
import glob
from PIL import Image
import pickle as pkl
from compute_PSNR_UP import *


path = '/scratch_net/ofsoundof/yawli/3D_appearance_dataset/Collection/Texture/x1'
path_sr = '/scratch_net/ofsoundof/yawli/3D_appearance_dataset/Collection/Texture_SR/x2_SR'

dataset = ['Collection'] #['MiddleBury', 'Collection', 'ETH3D', 'SyB3R']
psnr_all_y = []
psnr_all = []


for d in dataset:
    img_sr_list = glob.glob(os.path.join(path_sr, '*.png'))
    print('The images are: {}'.format(img_sr_list))
    psnr_ys = np.zeros([len(img_sr_list)+1])
    psnr_s = np.zeros([len(img_sr_list)+1])

    for i in range(len(img_sr_list)):
        img_sr_n = img_sr_list[i]
        name_img = os.path.splitext(os.path.basename(img_sr_n))[0]
        #print(name_img)
        
        img_sr = Image.open(img_sr_n)
        img_hr = Image.open(os.path.join(path, name_img + '.png'))
        #from IPython import embed; embed();
        w, h = img_sr.size
        s = 2
        img_hr_s = np.asarray(img_hr)[:h, :w, :]
        img_hr_s = shave(img_hr_s, s)
        img_sr_s = shave(np.asarray(img_sr), s)
        psnr_ys[i], psnr_s[i] = cal_pnsr_all(img_hr_s, img_sr_s)
        print('Image, {}: PSNR, {}; Difference, {}'.format(name_img, psnr_s[i], np.sum((img_hr_s - img_sr_s)**2)))

    psnr_ys[-1] = np.mean(psnr_ys[:-1], axis=0)
    psnr_s[-1] = np.mean(psnr_s[:-1], axis=0)
    psnr_all_y.append(psnr_ys)
    psnr_all.append(psnr_s)
    print(np.round(psnr_s,2))
