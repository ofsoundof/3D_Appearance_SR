__author__ = 'Yawei Li'
# This code generates Table 1 in the CVPR19 paper.
# The PSNR values are already pre-computed. This code just form the required value order.

from compute_PSNR_UP import *
import numpy as np

# save the html table. This is Table 1 used in the paper '3D appearance super-resolution with deep learning' CVPR2019.
def save_html_table(method, value, path):
    value = np.round(value, decimals=2).astype(str)
    value[4, 1::3] = '--'
    value[5, [4,7]] = '--'
    value[5, 0:3] = '--'
    value[5, 9:] = '--'
    content = value.tolist()
    content = [['& ' + content[i][j] for j in range(15)] for i in range(14)]
    # from IPython import embed; embed()

    for i in range(len(method)):
        content[i].insert(0, method[i])
    content.insert(0, ['Method'] + ['x2', 'x3', 'x4'] * 5)
    # from IPython import embed; embed()

    cont_str =  '\n'.join(html_table(content, 'Table 1: PSNR for all of the datasets'))
    with open(path, 'a') as f:
        f.write(cont_str)
        #save_file(cont_str, path)

# load the pickel file that contains the PSNR values for all of the scenes.
psnr = np.load('/home/yawli/projects/RCAN/RCAN_TrainCode/code/results/TA_PSNR.pkl')
psnr_rgb = psnr['psnr_all']
method_all = psnr['method']

psnr_sm = np.zeros((5, len(method_all), 3)) #dataset X method X scale
for i in range(4):
	# the average value for each subset
    psnr_sm[i, :, :] = psnr_rgb[i][-1, :, :]
    # compute the sum of PSNR values for the 24 scenes. 
    psnr_sm[-1, :, :] += np.sum(psnr_rgb[i][:-1, :, :], axis=0)
# compute the average value for all of the 24 scenes.
psnr_sm[-1, :, :] = psnr_sm[-1, :, :]/24
# refer to ./TA_Summary_RGB.html for the order of different methods.
psnr_table = np.ndarray((5, 14, 3))
psnr_table[:, 0:4, :] = psnr_sm[:, 0:4, :]
psnr_table[:, 4:6, :] = psnr_sm[:, -2:, :]
psnr_table[:, 6:14, :] = psnr_sm[:, [12,13,4,14,5,8,6,9], :]

# convert to the desired order in the CVPR paper.
table = np.reshape(np.transpose(psnr_table,(1,0,2)),(14,-1))
table_new_order = np.zeros(table.shape)
table_new_order[:, 0:3] = table[:, 6:9]
table_new_order[:, 3:6] = table[:, 3:6]
table_new_order[:, 6:9] = table[:, 0:3]
table_new_order[:, 9:] = table[:, 9:]
method = ['Nearest', 'Bilinear', 'Bicubic', 'Lanczos', 'HRST', 'HRST+', 'FSRCNN','SRRESNET','EDSR','RCAN', 'EDSR+', 'NLR-', 'NLR', 'NHR']
save_html_table(method, table_new_order, './results/table.html')




