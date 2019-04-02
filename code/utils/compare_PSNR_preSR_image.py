__author__ = 'yawli'

import numpy as np
import os
import glob
from PIL import Image
import pickle as pkl
from compute_PSNR_UP import *


path = '/scratch_net/ofsoundof/yawli/3D_appearance_dataset'

dataset = ['Collection'] #['MiddleBury', 'Collection', 'ETH3D', 'SyB3R']
subset = [['Buddha', 'Bunny', 'Fountain', 'Beethoven', 'Relief', 'Bird']]



for d in range(len(dataset)):
    D = dataset[d]
    psnr_all = []
    psnr_mean = np.zeros(len(subset[d]) + 1)
    for s in range(len(subset[d])):
        S = subset[d][s]
        img_hr_list = glob.glob(os.path.join(path, D, S, 'x1/Images/*.png'))
        img_sr_list = glob.glob(os.path.join(path, D, S, 'x2_RCAN/Images/*.png'))

        psnr_s = np.zeros([len(img_sr_list)+1])

        for i in range(len(img_sr_list)):
            img_sr_n = img_sr_list[i]
            img_hr_n = img_hr_list[i]
            
            img_sr = Image.open(img_sr_n)
            img_hr = Image.open(img_hr_n)
            #from IPython import embed; embed();
            w, h = img_sr.size
            crop = 2
            img_hr_s = np.asarray(img_hr)[:h, :w, :]
            img_hr_s = shave(img_hr_s, crop)
            img_sr_s = shave(np.asarray(img_sr), crop)
            _, psnr_s[i] = cal_pnsr_all(img_hr_s, img_sr_s)

        psnr_s[-1] = np.mean(psnr_s[:-1], axis=0)
        psnr_all.append(psnr_s)
        psnr_mean[s] = psnr_s[-1]
        #from IPython import embed; embed();
        print(psnr_s)
        print('The mean for {} in {} is {}.'.format(S, D, psnr_mean[s]))
    #print(psnr_all)
    psnr_mean[-1] = np.mean(psnr_mean[:-1])
    print(np.round(psnr_mean,2))
