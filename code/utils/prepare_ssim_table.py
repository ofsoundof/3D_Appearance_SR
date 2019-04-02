__author__ = 'Yawei Li'

# This code compute the SSIM value in the Supplementary Material of CVPR19 paper.
import numpy as np
import os
import glob
from PIL import Image
import copy
import pickle as pkl
from compute_PSNR_UP import *
from myssim import compare_ssim


if __name__ == '__main__':
    dataset = ['MiddleBury', 'Collection', 'ETH3D', 'SyB3R']
    path = '/scratch_net/ofsoundof/yawli/Datasets/texture_map'

    method = ['Nearest', 'Bilinear', 'Bicubic', 'Lanczos', 'HRST', 'HRST+', 'FSRCNN','SRRESNET','EDSR','RCAN', 'EDSR+', 'NLR-', 'NLR', 'NHR']
    len_m = len(method)
	
    img_dir = ['/home/yawli/Documents/3d-appearance-benchmark/SR/texture',
        '/scratch_net/ofsoundof/yawli/experiment/test/HRST+/EDSR_X{}_F256B32P{}E100_hr_{}_Input/results',
        '/scratch_net/ofsoundof/yawli/experiment/test/FSRCNN/x{}',
        '/scratch_net/ofsoundof/yawli/experiment/test/SRRESNET/SRRESNET_X{}_B16F64P{}/results',
        '/scratch_net/ofsoundof/yawli/experiment/test/EDSR_EDSR+_NLR_NLR-/EDSR_X{}_EDSR_TEST/results',
        '/scratch_net/ofsoundof/yawli/experiment/test/RCAN/RCAN_X{}/results',
        '/scratch_net/ofsoundof/yawli/experiment/test/EDSR_EDSR+_NLR_NLR-/EDSR_X{}_EDSR_CONTINUE/results',
        '/scratch_net/ofsoundof/yawli/experiment/test/EDSR_EDSR+_NLR_NLR-/EDSR_X{}_FINETUNE_EDSR_{}/results',
        '/scratch_net/ofsoundof/yawli/experiment/test/EDSR_EDSR+_NLR_NLR-/EDSR_X{}_FINETUNE_EDSR/results',
        '/scratch_net/ofsoundof/yawli/experiment/NLR_NHR_Config/EDSR_X{}_F256B32P{}E100_hr_{}/results'
        ]
    ssim_y = []
    ssim = []
    for d in dataset:
        img_hr_list = glob.glob(os.path.join(path, d, 'x1/Texture/*.png'))
        ssim_ys = np.zeros([len(img_hr_list)+1, len_m, 3])
        ssim_s = np.zeros([len(img_hr_list)+1, len_m, 3])

        for i in range(len(img_hr_list)):
            img_hr_n = img_hr_list[i]
            name_img = os.path.splitext(os.path.basename(img_hr_n))[0]
            img_hr = Image.open(img_hr_n)
            for s in range(2, 5):
                path_save = os.path.join(path, d, 'x{}'.format(s), 'sr')
                if not os.path.exists(path_save):
                    os.makedirs(path_save)
                img_lr_n = img_hr_n.replace('x1', 'x{}'.format(s))
                img_lr = Image.open(img_lr_n)
                w, h = img_lr.size

                img_hr.save(os.path.join(path_save, name_img + '_sr.png'))
                img_hr_s = np.array(img_hr)[:h*s, :w*s, :]
                img_hr_s = shave(img_hr_s, s)

                img_mr_near = img_lr.resize((w * s, h * s), Image.NEAREST)
                img_mr_bili = img_lr.resize((w * s, h * s), Image.BILINEAR)
                img_mr_bicu = img_lr.resize((w * s, h * s), Image.BICUBIC)
                img_mr_lanc = img_lr.resize((w * s, h * s), Image.LANCZOS)
                img_mr = [img_mr_near, img_mr_bili, img_mr_bicu, img_mr_lanc]
                #name_save = ['_nearest.png', '_bilinear.png', '_bicubic.png', '_lanczos.png']

                for m in range(4):
                    #img_mr[m].save(os.path.join(path_save, name_img + name_save[m]))
                    img_mr_s = shave(np.asarray(img_mr[m]), s)
                    #from IPython import embed; embed()
                    ssim_s[i, m, s-2] = compare_ssim(img_hr_s, img_mr_s, multichannel=True, gaussian_weights=True, use_sample_covariance=False)
                for m in range(4, len_m):

                    if method[m] == 'HRST':
                        name_img_sr = os.path.join(img_dir[m-4], d, 'x{}/{}.png'.format(s, name_img))
                    elif method[m] == 'HRST+':
                        name_img_sr1= os.path.join(img_dir[m-4].format(s, s*48, 'one'), name_img+'_x{}_FINETUNE_EDSR.png'.format(s))
                        name_img_sr2= os.path.join(img_dir[m-4].format(s, s*48, 'two'), name_img+'_x{}_FINETUNE_EDSR.png'.format(s))
                        name_img_sr = name_img_sr1 if os.path.exists(name_img_sr1) else name_img_sr2
                    elif method[m] == 'FSRCNN':
	                    name_img_sr = os.path.join(img_dir[m-4].format(s), name_img+'_FSRCNN.png')
                    elif method[m] == 'SRRESNET':
	                    name_img_sr = os.path.join(img_dir[m-4].format(s, s*24), name_img+'_x{}_SR_SRRESNET.png'.format(s))
                    elif method[m] == 'EDSR':
                        name_img_sr = os.path.join(img_dir[m-4].format(s), name_img+'_x{}_EDSR_TEST.png'.format(s))
                    elif method[m] == 'RCAN':
                        name_img_sr = os.path.join(img_dir[m-4].format(s), name_img+'_x{}_RCAN_RCAN.png'.format(s))
                    elif method[m] == 'EDSR+':
                        #from IPython import embed; embed()
                        name_img_sr = os.path.join(img_dir[m-4].format(s), name_img+'_x{}_EDSR_CONTINUE.png'.format(s))
                    elif method[m] == 'NLR-':
                        name_img_sr = os.path.join(img_dir[m-4].format(s, d), name_img+'_x{}_FINETUNE_EDSR.png'.format(s))
                    elif method[m] == 'NLR':
                        name_img_sr = os.path.join(img_dir[m-4].format(s), name_img+'_x{}_FINETUNE_EDSR.png'.format(s))
                    else:
                        name_img_sr1= os.path.join(img_dir[m-4].format(s,s*48,'one'), name_img+'_x{}_FINETUNE_EDSR.png'.format(s))
                        name_img_sr2= os.path.join(img_dir[m-4].format(s,s*48,'two'), name_img+'_x{}_FINETUNE_EDSR.png'.format(s))
                        name_img_sr = name_img_sr1 if os.path.exists(name_img_sr1) else name_img_sr2
                    if os.path.exists(name_img_sr):
                        print(name_img_sr)
                        img_sr = Image.open(name_img_sr)
                        w, h = img_sr.size
                        img_hr_s = np.asarray(img_hr)[:h, :w, :]
                        img_hr_s = shave(img_hr_s, s)
                        img_sr_s = shave(np.asarray(img_sr), s)
                        #from IPython import embed; embed()
                        ssim_s[i, m, s-2] = compare_ssim(img_hr_s, img_sr_s, multichannel=True, gaussian_weights=True, use_sample_covariance=False)

        ssim_s[-1, :, :] = np.mean(ssim_s[:-1, :, :], axis=0)
        ssim.append(ssim_s)
        save_html([os.path.splitext(os.path.basename(n))[0] for n in img_hr_list], method, ssim_s, d, './results/SSIM_RGB.html')

    with open('./results/SSIM.pkl', 'wb') as f:
        pkl.dump({'ssim': ssim, 'method': method}, f)

    #ssim_sm_y = np.zeros((5, len(method), 3))
    ssim_sm = np.zeros((5, len(method), 3))

    for i in range(len(dataset)):
        #ssim_sm_y[i, :, :] = ssim_y[i][-1, :, :]
        ssim_sm[i, :, :] = ssim[i][-1, :, :]
        #ssim_sm_y[-1, :, :] += np.sum(ssim_y[i][:-1, :, :], axis=0)
        ssim_sm[-1, :, :] += np.sum(ssim[i][:-1, :, :], axis=0)
    #ssim_sm_y[-1, :, :] = ssim_sm_y[-1, :, :]/24
    ssim_sm[-1, :, :] = ssim_sm[-1, :, :]/24
    #save_html(dataset, method, ssim_sm_y, 'All', './results/SSIM_Summary_Y.html')
    save_html(dataset, method, ssim_sm, 'All', './results/SSIM_Summary_RGB.html')
    
    
    def save_html_table(method, value, path):
        value = np.round(value, decimals=2).astype(str)
        value[4, 1::3] = '--'
        value[5, [4,7]] = '--'
        value[5, 0:3] = '--'
        value[5, 9:] = '--'
        content = value.tolist()
        content = [['& ' + content[i][j] for j in range(15)] for i in range(14)]
        for i in range(len(method)):
            content[i].insert(0, method[i])
        content.insert(0, ['Method'] + ['x2', 'x3', 'x4'] * 5)
        cont_str =  '\n'.join(html_table(content, 'Table 1: PSNR for all of the datasets'))
        with open(path, 'a') as f:
            f.write(cont_str)

    table = np.reshape(np.transpose(ssim_sm,(1,0,2)),(14,-1))
    table_new_order = np.zeros(table.shape)
    table_new_order[:, 0:3] = table[:, 6:9]
    table_new_order[:, 3:6] = table[:, 3:6]
    table_new_order[:, 6:9] = table[:, 0:3]
    table_new_order[:, 9:] = table[:, 9:]
    method = ['Nearest', 'Bilinear', 'Bicubic', 'Lanczos', 'HRST', 'HRST+', 'FSRCNN','SRRESNET','EDSR','RCAN', 'EDSR+', 'NLR-', 'NLR', 'NHR']
    save_html_table(method, table_new_order, './results/table_ssim.html')
    
    
    
