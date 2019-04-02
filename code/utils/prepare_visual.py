__author__ = 'Yawei Li'
# This code collects the super-resolved images of different methods in order for a better comparison between them.
# The collected images are used for the visual results.

import PIL.Image as Image
import os
import glob
method = ['EDSR', 'EDSR+', 'NLR-', 'NLR', 'NHR', 'HRST', 'HRST+']
#img_dir = #['/scratch_net/ofsoundof/yawli/Datasets/texture_map', #GT
img_dir = ['/scratch_net/ofsoundof/yawli/experiment/test/EDSR_EDSR+_NLR_NLR-/EDSR_X2_EDSR_TEST/results',
        '/scratch_net/ofsoundof/yawli/experiment/test/EDSR_EDSR+_NLR_NLR-/EDSR_X2_EDSR_CONTINUE/results',
        '/scratch_net/ofsoundof/yawli/experiment/test/EDSR_EDSR+_NLR_NLR-/EDSR_X2_FINETUNE_EDSR_{}/results',
        '/scratch_net/ofsoundof/yawli/experiment/test/EDSR_EDSR+_NLR_NLR-/EDSR_X2_FINETUNE_EDSR/results',
        '/scratch_net/ofsoundof/yawli/experiment/NLR_NHR_Config/EDSR_X2_F256B32P96E100_hr_{}/results',
        '/home/yawli/Documents/3d-appearance-benchmark/SR/texture',
        '/scratch_net/ofsoundof/yawli/experiment/test/HRST+/EDSR_X2_F256B32P96E100_hr_{}_Input/results']

len_m = len(method)

dataset = ['MiddleBury', 'Collection', 'ETH3D', 'SyB3R']
path = '/scratch_net/ofsoundof/yawli/Datasets/texture_map'

path_out = './results/SRimage'
for d in dataset:
    img_hr_list = glob.glob(os.path.join(path, d, 'x1/Texture/*.png'))
    for i in range(len(img_hr_list)):
        img_hr_n = img_hr_list[i]
        name_img = os.path.splitext(os.path.basename(img_hr_n))[0]
        print(name_img)
        img_hr = Image.open(img_hr_n)
        img_hr.save(os.path.join(path_out, name_img+'_1_GT.png'))
        img_lr_n = img_hr_n.replace('x1', 'x2')
        img_lr = Image.open(img_lr_n)
        w, h = img_lr.size
        img_nearest = img_lr.resize((w * 2, h * 2), Image.NEAREST)
        img_nearest.save(os.path.join(path_out, name_img+'_2_Nearest.png'))
        img_bilinear = img_lr.resize((w * 2, h * 2), Image.BILINEAR)
        img_bilinear.save(os.path.join(path_out, name_img+'_3_Bilinear.png'))
        img_lanczos = img_lr.resize((w * 2, h * 2), Image.LANCZOS)
        img_lanczos.save(os.path.join(path_out, name_img+'_4_Lanczos.png'))
        for m in range(len_m):
            if m == 0:
                name_img_sr = os.path.join(img_dir[m], name_img+'_x2_EDSR_TEST.png')
            elif m ==1:
                name_img_sr = os.path.join(img_dir[m], name_img+'_x2_EDSR_CONTINUE.png')
            elif m ==2:
                name_img_sr = os.path.join(img_dir[m].format(d), name_img+'_x2_FINETUNE_EDSR.png')
            elif m ==3:
                name_img_sr = os.path.join(img_dir[m], name_img+'_x2_FINETUNE_EDSR.png')
            elif m ==4:
                name_img_sr1= os.path.join(img_dir[m].format('one'), name_img+'_x2_FINETUNE_EDSR.png')
                name_img_sr2= os.path.join(img_dir[m].format('two'), name_img+'_x2_FINETUNE_EDSR.png')
                name_img_sr = name_img_sr1 if os.path.exists(name_img_sr1) else name_img_sr2
            elif m ==5:
                name_img_sr = os.path.join(img_dir[m], d, 'x2/{}.png'.format(name_img))
            elif m ==6:
                name_img_sr1= os.path.join(img_dir[m].format('one'), name_img+'_x2_FINETUNE_EDSR.png')
                name_img_sr2= os.path.join(img_dir[m].format('two'), name_img+'_x2_FINETUNE_EDSR.png')
                name_img_sr = name_img_sr1 if os.path.exists(name_img_sr1) else name_img_sr2
            
            if os.path.exists(name_img_sr):
                print(name_img_sr)
                img_sr = Image.open(name_img_sr)
                img_sr.save(os.path.join(path_out, name_img+'_{}_{}.png'.format(m+5, method[m])))

