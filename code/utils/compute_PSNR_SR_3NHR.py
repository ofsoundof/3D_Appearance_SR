__author__ = 'yawli'

import numpy as np
import os
import glob
from PIL import Image
import pickle as pkl
from compute_PSNR_UP import *

# compute PSNR value for NHR.
#path_sr = '/home/yawli/projects/RCAN/RCAN_TrainCode/experiment/test'
path_sr = '/scratch_net/ofsoundof/yawli/experiment/'
method_sr = ['EDSR']
len_m = len(method_sr)
train_flag = ['hr','lr','hr']
psnr_flag = ['_NORMAL_HR', '_NORMAL_LR', '_Res32']#, '_WN_Solo_E100']
len_f = len(psnr_flag)

dataset = ['MiddleBury', 'Collection', 'ETH3D', 'SyB3R']
path = '/scratch_net/ofsoundof/yawli/Datasets/texture_map'
psnr_all_y = []
psnr_all = []


with open('./results/TOTAL_PSNR.pkl', 'rb') as f:
    psnr = pkl.load(f)
method_all = psnr['method'] + [method_sr[m]+psnr_flag[f] for m in range(len_m) for f in range(len_f)]
num_pre = len(psnr['method'])

for d in dataset:
    img_hr_list = glob.glob(os.path.join(path, d, 'x1/Texture/*.png'))
    num = len_m * len_f
    psnr_ys = np.zeros([len(img_hr_list)+1, num_pre+num, 3])
    psnr_ys[:, :num_pre, :] = psnr['psnr_all_y'][dataset.index(d)]
    psnr_s = np.zeros([len(img_hr_list)+1, num_pre+num, 3])
    psnr_s[:, :num_pre, :] = psnr['psnr_all'][dataset.index(d)]

    for i in range(len(img_hr_list)):
        img_hr_n = img_hr_list[i]
        name_img = os.path.splitext(os.path.basename(img_hr_n))[0]
        print(name_img)
        img_hr = Image.open(img_hr_n)
        for s in range(2, 5):
            for m in range(len_m):
                for f in range(len_f):
                    flag = d + '_E100' if f == 2 else d
                    #from IPython import embed; embed()
                    tail = '_Res32' if f == 2 else ''
                    img_sr_n1 = os.path.join(path_sr, method_sr[m]+'_X{}_F256B32P{}E100_{}_one'.format(s,s*48,train_flag[f]) + tail, 'results',
                                           name_img+'_x{}_'.format(s)+'FINETUNE_EDSR.png')
                    img_sr_n2 = os.path.join(path_sr, method_sr[m]+'_X{}_F256B32P{}E100_{}_two'.format(s,s*48,train_flag[f]) + tail, 'results',
                                           name_img+'_x{}_'.format(s)+'FINETUNE_EDSR.png')
                    img_sr_n = img_sr_n1 if os.path.exists(img_sr_n1) else img_sr_n2
                    if os.path.exists(img_sr_n):
                        print(img_sr_n)
                        img_sr = Image.open(img_sr_n)
                        w, h = img_sr.size
                        img_hr_s = np.asarray(img_hr)[:h, :w, :]
                        img_hr_s = shave(img_hr_s, s)
                        img_sr_s = shave(np.asarray(img_sr), s)
                        psnr_ys[i, m*len_f+f+num_pre, s-2], psnr_s[i, m*len_f+f+num_pre, s-2] = cal_pnsr_all(img_hr_s, img_sr_s)

    psnr_ys[-1, :, :] = np.mean(psnr_ys[:-1, :, :], axis=0)
    psnr_s[-1, :, :] = np.mean(psnr_s[:-1, :, :], axis=0)
    psnr_all_y.append(psnr_ys)
    psnr_all.append(psnr_s)
    save_html([os.path.splitext(os.path.basename(n))[0] for n in img_hr_list], method_all, psnr_ys, d, './results/THL_PSNR_Y.html')
    save_html([os.path.splitext(os.path.basename(n))[0] for n in img_hr_list], method_all, psnr_s, d, './results/THL_PSNR_RGB.html')

with open('./results/THL_PSNR.pkl', 'wb') as f:
    pkl.dump({'psnr_all_y': psnr_all_y, 'psnr_all': psnr_all, 'method': method_all}, f)

psnr_sm_y = np.zeros((5, len(method_all), 3))
psnr_sm = np.zeros((5, len(method_all), 3))

for i in range(len(dataset)):
    psnr_sm_y[i, :, :] = psnr_all_y[i][-1, :, :]
    psnr_sm[i, :, :] = psnr_all[i][-1, :, :]
    psnr_sm_y[-1, :, :] += np.sum(psnr_all_y[i][:-1, :, :], axis=0)
    psnr_sm[-1, :, :] += np.sum(psnr_all[i][:-1, :, :], axis=0)
psnr_sm_y[-1, :, :] = psnr_sm_y[-1, :, :]/24
psnr_sm[-1, :, :] = psnr_sm[-1, :, :]/24

#psnr_sm_y[-1, -2:, :] = 0
#psnr_sm[-1, -2:, :] = 0

save_html(dataset, method_all, psnr_sm_y, 'All', './results/THL_Summary_Y.html')
save_html(dataset, method_all, psnr_sm, 'All', './results/THL_Summary_RGB.html')
