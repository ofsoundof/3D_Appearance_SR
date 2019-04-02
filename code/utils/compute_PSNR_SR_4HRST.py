__author__ = 'yawli'

import numpy as np
import os
import glob
from PIL import Image
import pickle as pkl
from compute_PSNR_UP import *

#path_sr = '/home/yawli/projects/RCAN/RCAN_TrainCode/experiment/test'
path_sr = '/scratch_net/ofsoundof/yawli/experiment/'
method_sr = ['FSRCNN', 'SRRESNET', 'RCAN', 'HRTM2014', 'HRTM2014_Post_Joint']#, 'HRTM2014_Post_Sep']
len_m = len(method_sr)
train_flag = ['']
psnr_flag = ['']
len_f = len(psnr_flag)

dataset = ['MiddleBury', 'Collection', 'ETH3D', 'SyB3R']
dataset_HRTM2014 = ['ColMapMiddlebury', 'Collection', 'ETH3D', 'SyB3R']
path = '/scratch_net/ofsoundof/yawli/Datasets/texture_map'
psnr_all_y = []
psnr_all = []


with open('./results/THL_PSNR.pkl', 'rb') as f:
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
                    #from IPython import embed; embed()
                    if method_sr[m] == 'FSRCNN':
                        img_sr_n = os.path.join(path_sr, 'test', method_sr[m], 'x{}'.format(s),
                                                name_img+'_{}.png'.format(method_sr[m]))
                    elif method_sr[m] == 'SRRESNET':
                        img_sr_n = os.path.join(path_sr, 'test', method_sr[m], method_sr[m]+'_X{}_B16F64P{}'.format(s,s*24), 'results',
                                                name_img+'_x{}_SR_{}.png'.format(s,method_sr[m]))
                    elif method_sr[m] == 'RCAN':
                        img_sr_n = os.path.join(path_sr, 'test', method_sr[m], method_sr[m]+'_X{}'.format(s), 'results',
                                                name_img+'_x{}_{}_{}.png'.format(s,method_sr[m],method_sr[m]))
                    elif method_sr[m] == 'HRTM2014':
                        img_sr_n = os.path.join('/home/yawli/Documents/3d-appearance-benchmark/SR/texture', d,
                                                'x{}/{}.png'.format(s,name_img))
                    else:
                        tail = '_' + d if method_sr[m] == 'HRTM2014_Post_Sep' else ''
                        img_sr_n1 = os.path.join(path_sr, 'test/HRST+/EDSR_X{}_F256B32P{}E100_hr_one_Input'.format(s,s*48) + tail, 'results',
                                               name_img+'_x{}_'.format(s)+'FINETUNE_EDSR.png')
                        img_sr_n2 = os.path.join(path_sr, 'test/HRST+/EDSR_X{}_F256B32P{}E100_hr_two_Input'.format(s,s*48) + tail, 'results',
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
    save_html([os.path.splitext(os.path.basename(n))[0] for n in img_hr_list], method_all, psnr_ys, d, './results/TA_PSNR_Y.html')
    save_html([os.path.splitext(os.path.basename(n))[0] for n in img_hr_list], method_all, psnr_s, d, './results/TA_PSNR_RGB.html')

with open('./results/TA_PSNR.pkl', 'wb') as f:
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

save_html(dataset, method_all, psnr_sm_y, 'All', './results/TA_Summary_Y.html')
save_html(dataset, method_all, psnr_sm, 'All', './results/TA_Summary_RGB.html')
