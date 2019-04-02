__author__ = 'yawli'

from model.edsr import EDSR
from model import common
import torch.nn as nn
import torch

def make_model(args, parent=False):
    m = args.submodel
    if m == 'HRST_CNN':
        return HRST_CNN(args)
    elif m == 'NLR':
        return NLR(args)
    elif m == 'NHR':
        return NHR(args)
    elif m == 'NHR_Res32':
        return NHR_Res32(args)
    else:
        NotImplementedError('The architecture {} is not implemented.'.format(m))

class NHR_Res32(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(NHR_Res32, self).__init__()
        n_resblocks = args.n_resblocks
        args.n_resblocks = args.n_resblocks - args.n_resblocks_ft
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        tail_ft2 = [
            common.ResBlock(
                conv, n_feats+4, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(args.n_resblocks_ft)
        ]
        tail_ft2.append(conv(n_feats+4, args.n_colors, kernel_size))

        tail_ft1 = [
            common.Upsampler(conv, scale, n_feats, act=False),
        ]
        premodel = EDSR(args)
        self.sub_mean = premodel.sub_mean
        self.head = premodel.head
        body = premodel.body
        body_child = list(body.children())
        body_ft = [body_child.pop()]
        self.body = nn.Sequential(*body_child)
        self.body_ft = nn.Sequential(*body_ft)
        self.tail_ft1 = nn.Sequential(*tail_ft1)
        self.tail_ft2 = nn.Sequential(*tail_ft2)
        self.add_mean = premodel.add_mean
        args.n_resblocks = n_resblocks
        # self.premodel = EDSR(args)
        # from IPython import embed; embed(); exit()
    def forward(self, inputs):
        x = inputs[0]
        y = inputs[1]
        # z = inputs[2]
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res = self.body_ft(res)
        # print(res.shape)
        # print(y.shape)
        res += x
        x = self.tail_ft1(x)
        x = torch.cat((x, y), dim=1)
        #res = self.body_ft(res)
        #res += x

        x = self.tail_ft2(x)
        x = self.add_mean(x)
        return x
        # return self.premodel(inputs[0])

class HRST_CNN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(HRST_CNN, self).__init__()
        n_resblocks = args.n_resblocks
        args.n_resblocks = args.n_resblocks - args.n_resblocks_ft
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        body_ft = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(args.n_resblocks_ft)
        ]
        body_ft.append(conv(n_feats, n_feats, kernel_size))

        tail_ft = [
            conv(n_feats, n_feats, kernel_size),
            conv(n_feats, n_feats, kernel_size),
            conv(n_feats, args.n_colors, kernel_size)
        ]
        premodel = EDSR(args)
        self.sub_mean = premodel.sub_mean
        self.head = premodel.head
        body = premodel.body
        body_child = list(body.children())
        body_child.pop()
        self.body = nn.Sequential(*body_child)
        self.body_ft = nn.Sequential(*body_ft)
        self.tail_ft = nn.Sequential(*tail_ft)
        self.add_mean = premodel.add_mean
        args.n_resblocks = n_resblocks
        # self.premodel = EDSR(args)
        # from IPython import embed; embed(); exit()
    def forward(self, inputs):
        x = inputs[0]
        #y = inputs[1]
        # z = inputs[2]
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        # print(res.shape)
        # print(y.shape)
        #res = torch.cat((res, y), dim=1)
        res = self.body_ft(res)
        res += x

        x = self.tail_ft(res)
        x = self.add_mean(x)
        return x
        # return self.premodel(inputs[0])

class NLR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(NLR, self).__init__()
        n_resblocks = args.n_resblocks
        args.n_resblocks = args.n_resblocks - args.n_resblocks_ft
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        body_ft = [
            common.ResBlock(
                conv, n_feats+4, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(args.n_resblocks_ft)
        ]
        body_ft.append(conv(n_feats+4, n_feats, kernel_size))

        tail_ft = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]
        premodel = EDSR(args)
        self.sub_mean = premodel.sub_mean
        self.head = premodel.head
        body = premodel.body
        body_child = list(body.children())
        body_child.pop()
        self.body = nn.Sequential(*body_child)
        self.body_ft = nn.Sequential(*body_ft)
        self.tail_ft = nn.Sequential(*tail_ft)
        self.add_mean = premodel.add_mean
        args.n_resblocks = n_resblocks
        # self.premodel = EDSR(args)
        # from IPython import embed; embed(); exit()
    def forward(self, inputs):
        x = inputs[0]
        y = inputs[1]
        # z = inputs[2]
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        # print(res.shape)
        # print(y.shape)
        res = torch.cat((res, y), dim=1)
        res = self.body_ft(res)
        res += x

        x = self.tail_ft(res)
        x = self.add_mean(x)
        return x
        # return self.premodel(inputs[0])

class NHR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(NHR, self).__init__()
        n_resblocks = args.n_resblocks
        args.n_resblocks = args.n_resblocks - args.n_resblocks_ft
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        n_color = args.n_colors
        self.normal_lr = args.normal_lr == 'lr'
        self.args = args
        if self.normal_lr:
            body_ft = [
                ResBlock(
                    conv, n_feats+4, n_feats+4, kernel_size, act=act, res_scale=args.res_scale
                ) for _ in range(args.n_resblocks_ft)
            ]
            body_ft.append(conv2d(n_feats+4, n_feats, kernel_size, act=True))

            tail_ft1 = [
                common.Upsampler(conv, scale, n_feats, act=True),
                conv2d(n_feats, n_feats+4, kernel_size, act=True),
            ]
            tail_ft2 = [conv2d(n_feats+4, n_feats+4, kernel_size, act=True),
                        conv2d(n_feats+4, n_feats+4, kernel_size, act=True),
                        conv2d(n_feats+4, n_feats+4, kernel_size, act=True),
                        conv2d(n_feats+4, n_color, kernel_size, act=False)]
        else:
            body_ft = [ResBlock(conv, n_feats, n_feats, kernel_size, act=act, res_scale=args.res_scale),
                       ResBlock(conv, n_feats, n_feats, kernel_size, act=act, res_scale=args.res_scale)]
                # ResBlock(conv, n_feats+4, n_feats+4, kernel_size, act=act, res_scale=args.res_scale)
            #]
            body_ft.append(conv2d(n_feats, n_feats, kernel_size, act=True))

            tail_ft1 = [
                common.Upsampler(conv, scale, n_feats, act=True),
                conv2d(n_feats, n_feats, kernel_size, act=True),
            ]
            tail_ft2 = [conv2d(n_feats+4, n_feats+4, kernel_size, act=True),
                        conv2d(n_feats+4, n_feats+4, kernel_size, act=True),
                        conv2d(n_feats+4, n_feats+4, kernel_size, act=True),
                        conv2d(n_feats+4, n_color, kernel_size, act=False)]

        premodel = EDSR(args)
        self.sub_mean = premodel.sub_mean
        self.head = premodel.head
        body = premodel.body
        body_child = list(body.children())
        body_child.pop()
        self.body = premodel.body
        self.body_ft = nn.Sequential(*body_ft)
        self.tail_ft1 = nn.Sequential(*tail_ft1)
        self.tail_ft2 = nn.Sequential(*tail_ft2)
        self.add_mean = premodel.add_mean
        args.n_resblocks = n_resblocks
        # self.premodel = EDSR(args)
        # from IPython import embed; embed(); exit()
    def forward(self, inputs):
        x = inputs[0]
        y = inputs[1]
        # z = inputs[2]
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        if self.normal_lr: res = torch.cat((res, y), dim=1)
        res = self.body_ft(res)
        res += x
        x = self.tail_ft1(res)
        # from IPython import embed; embed(); exit()
        if not self.normal_lr: x = torch.cat((x, y), dim=1)
        x = self.tail_ft2(x)
        x = self.add_mean(x)
        return x
        # return self.premodel(inputs[0])

class ResBlock(nn.Module):
    def __init__(self, conv, in_feat, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            feat = in_feat if i == 0 else n_feat
            m.append(conv(feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

def conv2d(in_feat, out_feat, kernel_size, bias=True, act=True):
    c = [nn.Conv2d(in_feat, out_feat, kernel_size, stride=1, padding=(kernel_size//2), bias=bias)]
    if act: c.append(nn.ReLU())
    return nn.Sequential(*c)
