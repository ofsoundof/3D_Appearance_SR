__author__ = 'yawli'

import numpy as np
import os
import glob
from PIL import Image
import copy
import pickle as pkl

def shave(img, scale):
    return img[scale:-scale, scale:-scale, :]

def cal_pnsr(img_hr, img_mr, mask):
    '''
    compute psnr value. Black regions are excluded from the computing.
    '''
    mask_sum = np.sum(mask)
    mask_sum = mask_sum * 3 if img_hr.ndim == 3 else mask_sum
    mse = np.sum(np.square(img_hr - img_mr))/mask_sum
    psnr = 10 * np.log10(255**2/mse)
    return psnr

def cal_pnsr_all(img_hr, img_mr):
    img_hr_y = rgb2ycbcr(img_hr).astype(np.float32)
    img_mr_y = rgb2ycbcr(img_mr).astype(np.float32)
    mask = (img_hr_y != 16).astype(np.float32)
    img_hr = img_hr.astype(np.float32)
    img_mr = img_mr.astype(np.float32)
    # from IPython import embed; embed(); exit()
    psnr_y = cal_pnsr(img_hr_y, img_mr_y, mask)
    psnr = cal_pnsr(img_hr, img_mr, mask)
    return psnr_y, psnr

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img = img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def save_file(cont, path):
    with open(path, 'a') as f:
        f.write(cont)

def html_table(lol, cap):
    yield '<table border="1">'
    yield '  <caption>{}</caption>'.format(cap)
    for sublist in lol:
        yield '  <tr>'
        yield '    <td>' + '</td>  <td>'.join(sublist) + '</td>'
        yield '  </tr>'
    yield '</table>'

def save_html_tbl(img_name, test_res, opt, path):
    img_name.insert(0, 'Average')
    content = np.round(test_res, decimals=4).astype(str).tolist()
    content.insert(0, img_name)
    row_name = ['', 'PSNR', 'SSIM', 'Runtime']
    for i in range(len(row_name)):
        content[i].insert(0, row_name[i])
    cont_str =  '\n'.join(html_table(content, 'Table 1: PSNR/SSIM/Runtime for {} on {}'.format(opt.ckpt_name, opt.test_set)))
    path += '/PSNR_SSIM_Runtime_{}.html'.format(opt.test_set)
    save_file(cont_str, path)

def save_html(img_name, method, value, set, path):
    value = np.round(value, decimals=2).astype(str)
    img_name = copy.copy(img_name)
    img_name.append('Average')
    _, _, scale = value.shape
    content = []
    for s in range(scale):
        content_s = value[:, :, s].tolist()
        # from IPython import embed; embed()
        for i in range(len(img_name)):
            content_s[i].insert(0, 'x{}'.format(s+2))
            content_s[i].insert(0, img_name[i])

        content.extend(content_s)
        content.append([''.join(['-'] * 10)] * (len(method) + 2))

    method = ['Image', 'Scale'] + method
    content.insert(0, method)

    cont_str =  '\n'.join(html_table(content, 'Table 1: PSNR for dataset {}'.format(set)))
    save_file(cont_str, path)

if __name__ == '__main__':
    dataset = ['MiddleBury', 'Collection', 'ETH3D', 'SyB3R']
    path = '/scratch_net/ofsoundof/yawli/Datasets/texture_map'

    method = ['Nearest', 'Bilinear', 'Bicubic', 'Lanczos']
    psnr_y = []
    psnr = []

    for d in dataset:
        img_hr_list = glob.glob(os.path.join(path, d, 'x1/Texture/*.png'))
        psnr_ys = np.zeros([len(img_hr_list)+1, 4, 3])
        psnr_s = np.zeros([len(img_hr_list)+1, 4, 3])

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
                name_save = ['_nearest.png', '_bilinear.png', '_bicubic.png', '_lanczos.png']

                for m in range(4):
                    img_mr[m].save(os.path.join(path_save, name_img + name_save[m]))
                    img_mr_s = shave(np.asarray(img_mr[m]), s)
                    psnr_ys[i, m, s-2], psnr_s[i, m, s-2] = cal_pnsr_all(img_hr_s, img_mr_s)

        psnr_ys[-1, :, :] = np.mean(psnr_ys[:-1, :, :], axis=0)
        psnr_s[-1, :, :] = np.mean(psnr_s[:-1, :, :], axis=0)
        psnr_y.append(psnr_ys)
        psnr.append(psnr_s)
        save_html([os.path.splitext(os.path.basename(n))[0] for n in img_hr_list], method, psnr_ys, d, './results/UP_PSNR_Y.html')
        save_html([os.path.splitext(os.path.basename(n))[0] for n in img_hr_list], method, psnr_s, d, './results/UP_PSNR_RGB.html')

    with open('./results/UP_PSNR.pkl', 'wb') as f:
        pkl.dump({'psnr_up_y': psnr_y, 'psnr_up': psnr}, f)

    psnr_sm_y = np.zeros((5, len(method), 3))
    psnr_sm = np.zeros((5, len(method), 3))

    for i in range(len(dataset)):
        psnr_sm_y[i, :, :] = psnr_y[i][-1, :, :]
        psnr_sm[i, :, :] = psnr[i][-1, :, :]
        psnr_sm_y[-1, :, :] += np.sum(psnr_y[i][:-1, :, :], axis=0)
        psnr_sm[-1, :, :] += np.sum(psnr[i][:-1, :, :], axis=0)
    psnr_sm_y[-1, :, :] = psnr_sm_y[-1, :, :]/24
    psnr_sm[-1, :, :] = psnr_sm[-1, :, :]/24
    save_html(dataset, method, psnr_sm_y, 'All', './results/UP_Summary_Y.html')
    save_html(dataset, method, psnr_sm, 'All', './results/UP_Summary_RGB.html')
