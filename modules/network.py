import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from math import pi
import numpy as np
import torch
import os
import csv
import math
import random
from tqdm import tqdm

from PIL import Image
from argparse import ArgumentParser
import torch.nn as nn

from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor, ToPILImage
from scipy import io
#from modules.layers import ConvOffset2D

import math 
import numpy as np

#outs = tanh(ylogit), outc = tanh(xlogit)) with a loss function 0.5((sin(pred) - outs)^2 + (cos(pred) - outc)^2


class UNetUp(nn.Module):

    def __init__(self, in_channels, features, out_channels, is_bn = True):
        super().__init__()
        if is_bn:
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, features, 3, padding = 1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, 3, padding = 1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(features, out_channels, 2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, features, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(features, out_channels, 2, stride=2),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.up(x)

class UNetUpTanh(nn.Module):

    def __init__(self, in_channels, features, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding = 1),
            nn.Tanh(),
            nn.Conv2d(features, features, 3, padding = 1),
            nn.Tanh(),
            nn.ConvTranspose2d(features, out_channels, 2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        return self.up(x)

class UNetDown(nn.Module):

    def __init__(self, in_channels, out_channels, is_bn = True):
        super().__init__()

        if is_bn:
            self.down = nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(in_channels, out_channels, 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.down = nn.Sequential(
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(in_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.down(x)

class UNetDownTanh(nn.Module):

    def __init__(self, in_channels, out_channels, is_bn = True):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels, out_channels, 3, padding = 1),
            nn.Tanh(),
            nn.Conv2d(out_channels, out_channels, 3, padding = 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.down(x)
        
class UNet_radius_center_denseconv(nn.Module):

    def __init__(self, num_classes, num_channels, num_params=6, width=4, n_convs=10, is_bn = True, dense_bn = True):
        super().__init__()
        self.n_convs = n_convs
        self.is_bn = is_bn
        self.dense_bn = dense_bn
        if self.is_bn:
            self.first = nn.Sequential(
                nn.Conv2d(num_channels, width, 3, padding = 1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding = 1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True)
            )
        else:
            self.first = nn.Sequential(
                nn.Conv2d(num_channels, width, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding = 1),
                nn.ReLU(inplace=True)
            )
            
        self.dec2 = UNetDown(width, width*2, is_bn = self.is_bn)
        self.dec3 = UNetDown(width*2, width*4, is_bn = self.is_bn)
        self.dec4 = UNetDown(width*4, width*8, is_bn = self.is_bn)

        self.center = nn.MaxPool2d(2, stride=2)
        self.center1 = nn.Conv2d(width*8, width*16, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(width*16)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(width*16, width*16, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(width*16)
        self.center2relu = nn.ReLU(inplace=True)
        self.center3 = nn.ConvTranspose2d(width*16, width*8, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(width*8)
        self.center3_relu = nn.ReLU(inplace=True)
        
        #self.center4 = nn.AdaptiveAvgPool2d((1,1)
        #self.center4_relu = nn.ReLU(inplace=True)
        self.xyr_convs = []
        self.xyr_bns = []
        self.xyr_relus = []
        
        for i in range(self.n_convs):
            self.xyr_convs.append(nn.Conv2d(width*16*(i+1), width*16, 3, padding = 1))
            self.xyr_bns.append(nn.BatchNorm2d(width*16))
            self.xyr_relus.append(nn.ReLU(inplace=True))
        
        self.xyr_convs = nn.ModuleList(self.xyr_convs)
        self.xyr_bns = nn.ModuleList(self.xyr_bns)
        self.xyr_relus = nn.ModuleList(self.xyr_relus)
        
        
        # 64 x 20 x 15
        self.xyr_input = nn.Flatten()
        self.xyr_linear = nn.Linear(width*16 * 20 * 15, num_params)
        
        self.enc4 = UNetUp(width*16, width*8, width*4, is_bn = self.is_bn)
        self.enc3 = UNetUp(width*8, width*4, width*2, is_bn = self.is_bn)
        self.enc2 = UNetUp(width*4, width*2, width, is_bn = self.is_bn)
        if self.is_bn:
            self.enc1 = nn.Sequential(
                nn.Conv2d(width*2, width, 3, padding = 1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding = 1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
            )
        else:
            self.enc1 = nn.Sequential(
                nn.Conv2d(width*2, width, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding = 1),
                nn.ReLU(inplace=True),
            )
        self.final = nn.Conv2d(width, num_classes, 1)
        self._initialize_weights()
     
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                m.bias.data.zero_()


    def forward(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr = [center7]
        dense_col = [center7]
        for i in range(self.n_convs):
            xyr.append(self.xyr_convs[i](torch.cat(dense_col, 1)))
            if self.dense_bn:
                xyr.append(self.xyr_bns[i](xyr[-1]))
            dense_col.append(self.xyr_relus[i](xyr[-1]))
                
        
        xyr_lin1 = self.xyr_input(dense_col[-1])
        xyr_lin2 = self.xyr_linear(xyr_lin1)

        enc4 = self.enc4(torch.cat([center10, dec4], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1], 1))

        return self.final(enc1), xyr_lin2
        
    def encode_params(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        xyr = [center7]
        dense_col = [center7]
        for i in range(self.n_convs):
            xyr.append(self.xyr_convs[i](torch.cat(dense_col, 1)))
            if self.dense_bn:
                xyr.append(self.xyr_bns[i](xyr[-1]))
            dense_col.append(self.xyr_relus[i](xyr[-1]))
                
        
        xyr_lin1 = self.xyr11_input(dense_col[-1])
        xyr_lin2 = self.xyr11_linear(xyr_lin1)

        return xyr_lin2

class UNet(nn.Module):

    def __init__(self, num_classes, num_channels, width=4):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(num_channels, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.dec2 = UNetDown(width, 2*width)
        self.dec3 = UNetDown(2*width, 4*width)
        self.dec4 = UNetDown(4*width, 8*width)      


        self.center = nn.MaxPool2d(2, stride=2)
        self.center1 = nn.Conv2d(8*width, 16*width, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(16*width)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(16*width, 16*width, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(16*width)
        self.center2relu = nn.ReLU(inplace=True)
        self.center3 = nn.ConvTranspose2d(16*width, 8*width, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(8*width)
        self.center3_relu = nn.ReLU(inplace=True)
        
        
        
        self.enc4 = UNetUp(16*width, 8*width, 4*width)
        self.enc3 = UNetUp(8*width, 4*width, 2*width)
        self.enc2 = UNetUp(4*width, 2*width, width)
        self.enc1 = nn.Sequential(
            nn.Conv2d(2*width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(width, num_classes, 1)
        self._initialize_weights()
     
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1 = self.center(dec4)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)

        enc4 = self.enc4(torch.cat([center10, dec4], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1], 1))

        return self.final(enc1)


class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):
        if (x if x is not None else img).device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------


class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

    def extra_repr(self):
        return f'group_size={self.group_size}, num_channels={self.num_channels:d}'

#----------------------------------------------------------------------------


class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------


class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, update_emas=False, **block_kwargs):
        _ = update_emas # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'
            
        

class SIFLayerMask(nn.Module):
    def __init__(self, polar_height, polar_width, filter_mat, device, channels = 1):
        super().__init__()
        self.device = device
        self.polar_height = polar_height
        self.polar_width = polar_width
        self.channels = channels
        self.angles = angles = np.arange(0, 2 * np.pi, 2 * np.pi / self.polar_width)
        self.cos_angles = np.zeros((self.polar_width))
        self.sin_angles = np.zeros((self.polar_width))
        for i in range(self.polar_width):
            self.cos_angles[i] = np.cos(self.angles[i])
            self.sin_angles[i] = np.sin(self.angles[i])
        assert filter_mat.shape[0] == filter_mat.shape[1]
        self.filter_size = filter_mat.shape[0]
        self.num_filters = filter_mat.shape[2]
        self.filter = torch.FloatTensor(filter_mat).requires_grad_(False)
        #print(self.filter)
        self.filter = torch.rot90(self.filter, 2, [0, 1])
        self.filter = torch.moveaxis(self.filter.unsqueeze(0), 3, 0)
        self.filter = torch.cat([self.filter] * self.channels, dim=1)
       
    def grid_sample(self, input, grid):
        # grid: [-1, 1]
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1).to(self.device)
        return torch.nn.functional.grid_sample(input, newgrid)

    def cartToPol(self, image, mask, pupil_xyr, iris_xyr):
        
        batch_size = image.shape[0]
        width = image.shape[3]
        height = image.shape[2]

        polar_height = self.polar_height
        polar_width = self.polar_width

        pupil_xyr = pupil_xyr.clone().detach().requires_grad_(False)
        iris_xyr = iris_xyr.clone().detach().requires_grad_(False)
        
        theta = (2*pi*torch.linspace(1,polar_width,polar_width)/polar_width).requires_grad_(False)
        pxCirclePoints = pupil_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width) #b x 512
        pyCirclePoints = pupil_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
        
        ixCirclePoints = iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)  #b x 512
        iyCirclePoints = iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512

        radius = (torch.linspace(0,polar_height,polar_height)/polar_height).reshape(-1, 1).requires_grad_(False)  #64 x 1
        
        pxCoords = torch.matmul((1-radius), pxCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        pyCoords = torch.matmul((1-radius), pyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        
        
        ixCoords = torch.matmul(radius, ixCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
        iyCoords = torch.matmul(radius, iyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512

        x = torch.clamp(pxCoords + ixCoords, 0, width-1).float()
        x_norm = (x/(width-1))*2 - 1 #b x 64 x 512

        y = torch.clamp(pyCoords + iyCoords, 0, height-1).float()
        y_norm = (y/(height-1))*2 - 1  #b x 64 x 512

        grid_sample_mat = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1)

        image_polar = self.grid_sample(image, grid_sample_mat)
        image_polar = torch.clamp(torch.round(image_polar), min=0, max=255)
        if mask is not None:
            mask_mean = (mask.max() + mask.min())/2
            mask[mask<=mask_mean] = 0
            mask[mask>mask_mean] = 255
            mask_polar = self.grid_sample(torch.tensor(mask).float(), grid_sample_mat)
            mask_polar = np.uint8(np.clip(np.around(mask_polar.clone().detach().cpu().numpy()), 0, 255))
        else:
            mask_polar = None

        return image_polar, mask_polar

    def getCodes(self, image_polar):
        r = int(np.floor(self.filter_size / 2))
        imgWrap = Variable(torch.zeros((image_polar.shape[0], image_polar.shape[1], r*2+self.polar_height, r*2+self.polar_width)).requires_grad_(True))
        
        imgWrap[:, :, :r, :r] = torch.clone(image_polar[:, :, -r:, -r:])
        imgWrap[:, :, :r, r:-r] = torch.clone(image_polar[:, :, -r:, :])
        imgWrap[:, :, :r, -r:] = torch.clone(image_polar[:, :, -r:, :r])

        imgWrap[:, :, r:-r, :r] = torch.clone(image_polar[:, :, :, -r:])
        imgWrap[:, :, r:-r, r:-r] = torch.clone(image_polar)
        imgWrap[:, :, r:-r, -r:] = torch.clone(image_polar[:, :, :, :r])

        imgWrap[:, :, -r:, :r] = torch.clone(image_polar[:, :, :r, -r:])
        imgWrap[:, :, -r:, r:-r] = torch.clone(image_polar[:, :, :r, :])
        imgWrap[:, :, -r:, -r:] = torch.clone(image_polar[:, :, :r, :r])

        codes = nn.functional.conv2d(imgWrap, self.filter, stride=1, padding=0)
        
        return codes

    def forward(self, image, pupil_xyr, iris_xyr, mask=None):
        image_polar, mask_polar = self.cartToPol(image, mask, pupil_xyr, iris_xyr)
        codes = self.getCodes(image_polar)       
        return codes, image_polar, mask_polar