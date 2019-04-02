__author__ = 'yawli'
import math
import torch.nn as nn
import torch
from model import common
# from model import ops

def make_model(args, parent=False):
    return SRResNet(args)

def norm(norm_type, channel, group):
    if norm_type == 'batchnorm':
        norm = nn.BatchNorm2d(channel)
    elif norm_type == 'groupnorm':
        norm = nn.GroupNorm(group, channel)
    elif norm_type == 'instancenorm':
        norm = nn.InstanceNorm2d(channel)
    elif norm_type == 'instancenorm_affine':
        norm = nn.InstanceNorm2d(channel, affine=True)
    elif norm_type == 'layernorm':
        norm = nn.LayerNorm(channel)
    else:
        norm = None
    return norm

class VarBlockSimple(nn.Module):

    def __init__(self, conv=common.default_conv, n_feats=64, kernel_size=3, reg_act=nn.Softplus(), rescale=1, norm_f=None):
        super(VarBlockSimple, self).__init__()
        if norm_f is not None:
            conv_mask = [norm_f, nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=kernel_size//2, groups=n_feats), reg_act]
        else:
            conv_mask = [nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=kernel_size//2, groups=n_feats), reg_act]
        conv_body = [conv(n_feats, n_feats, kernel_size), nn.PReLU()]
        self.rescale = rescale
        self.conv_mask = nn.Sequential(*conv_mask)
        self.conv_body = nn.Sequential(*conv_body)

    def forward(self, x):
        res = self.conv_body(self.conv_mask(x) * x)
        x = res.mul(self.rescale) + x
        return x

class JointAttention(nn.Module):

    def __init__(self, conv=common.default_conv, n_feats=64, kernel_size=3, reg_act=nn.Softplus(), rescale=1, norm_f=None):
        super(JointAttention, self).__init__()
        mask_conv = [nn.Conv2d(n_feats, 16, kernel_size=kernel_size, stride=4, padding=kernel_size//2), nn.PReLU()]
        mask_deconv = nn.ConvTranspose2d(16, n_feats, kernel_size=kernel_size, stride=4, padding=1)
        mask_deconv_act = nn.Softmax2d()
        conv_body = [conv(n_feats, n_feats, kernel_size), nn.PReLU()]
        self.mask_conv = nn.Sequential(*mask_conv)
        self.mask_deconv = mask_deconv
        self.mask_deconv_act = mask_deconv_act
        # self.ca = CALayer(n_feats)
        self.conv_body = nn.Sequential(*conv_body)

    def forward(self, x):
        mask = self.mask_deconv_act(self.mask_deconv(self.mask_conv(x), output_size=x.size()))
        res = mask * x
        # res = self.ca(res)
        res = self.conv_body(res)
        x = res + x
        return x

class UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels, scale, multi_scale,
                 group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up =  _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self,
				 n_channels, scale,
				 group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=group), nn.PReLU()]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1, groups=group), nn.PReLU()]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        out = self.body(x)
        return out


class SRResNet(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SRResNet, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        # scale = args.scale[0]
        act = nn.PReLU()

        multi_scale = len(args.scale) > 1
        self.scale_idx = 0
        scale = args.scale[self.scale_idx]
        group = 1
        self.scale = args.scale

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        norm_f = norm(args.norm_type, args.n_feats, args.n_groups)
        act_vconv = common.act_vconv(args.res_act)

        head = [conv(args.n_colors, n_feats, kernel_size), act]
        body_r = [JointAttention(conv, n_feats, kernel_size, reg_act=act_vconv, norm_f=norm_f, rescale=args.res_scale)
                         for _ in range(n_resblocks)]
        #body_r = [common.ResBlock(conv, n_feats, kernel_size, bn=False, act=act, res_scale=args.res_scale, num_conv=2)
        #                 for _ in range(n_resblocks)]


        body_conv = [conv(n_feats, n_feats, kernel_size)]
        #body_conv = [conv(n_feats, n_feats, kernel_size), nn.BatchNorm2d(n_feats)]

        # tail = [
        #     common.Upsampler(conv, scale, n_feats, act=act),
        #     conv(n_feats, args.n_colors, kernel_size)
        # ]

        tail = UpsampleBlock(n_feats,
                             scale=scale,
                             multi_scale=multi_scale,
                             group=group)
        tail_conv = [conv(n_feats, args.n_colors, kernel_size)]

        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.head = nn.Sequential(*head)
        self.body_r = nn.Sequential(*body_r)
        self.body_conv = nn.Sequential(*body_conv)
        self.tail = tail
        self.tail_conv = nn.Sequential(*tail_conv)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        f = self.body_r(x)
        f = self.body_conv(f)
        scale = self.scale[self.scale_idx]
        x = self.tail(x + f, scale)
        x = self.tail_conv(x)
        x = self.add_mean(x)
        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
