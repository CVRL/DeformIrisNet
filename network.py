import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import models, transforms
from math import pi
import numpy as np
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
import scipy
from scipy import io
#from modules.layers import ConvOffset2D

import math 
import numpy as np
import lpips

import numpy as np
import torch
#outs = tanh(ylogit), outc = tanh(xlogit)) with a loss function 0.5((sin(pred) - outs)^2 + (cos(pred) - outc)^2

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.NLLLoss(weight)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, outputs, targets):
        return self.loss(self.softmax(outputs), targets)

class HingeLoss(nn.Module):
    
    def __init__(self, device, p=1):
        super().__init__()
        self.p = p
        self.device = device
    
    def forward(self, outputs, targets):
        loss = torch.pow(torch.max(torch.tensor(0.).to(self.device), 1 - targets * outputs), self.p)
        return torch.mean(loss)

class HingeLossWithSoftLabels(nn.Module):
    def __init__(self, device, label_smoothing=0.2):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.device = device
        self.hinge_loss = HingeLoss(device, p=1)
    def forward(self, input, target):
        zero = torch.zeros(target.shape).to(self.device)
        noise_neg = torch.rand(target.shape).to(self.device)*self.label_smoothing
        noise_neg = torch.where(target < 0, noise_neg, zero)
        noise_pos = torch.rand(target.shape).to(self.device)*self.label_smoothing
        noise_pos = torch.where(target > 0, noise_pos, zero)
        target_soft = target + noise_neg.requires_grad_(False) - noise_pos.requires_grad_(False)
        return self.hinge_loss(input, target_soft)
        

class UNetUpConv(nn.Module):

    def __init__(self, in_channels, features, out_channels, is_bn = True):
        super().__init__()
        if is_bn:
            self.process = nn.Sequential(
                nn.Conv2d(in_channels, features, 3, padding = 1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            )
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(features, features, 2, stride=2),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            )
            self.final = nn.Sequential(
                nn.Conv2d(features * 2, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.process = nn.Sequential(
                nn.Conv2d(in_channels, features, 3, padding = 1),
                nn.ReLU(inplace=True),
            )
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(features, features, 2, stride=2),
                nn.ReLU(inplace=True)
            )
            self.final = nn.Sequential(
                nn.Conv2d(features * 2, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        xp = self.process(x)
        xi = F.interpolate(xp, scale_factor=2, mode='nearest')
        xc = self.upsample(xp)
        xr = self.final(torch.cat([xi, xc], 1))        
        return xr

class UNetDownConv(nn.Module):

    def __init__(self, in_channels, out_channels, is_bn = True):
        super().__init__()

        if is_bn:
            self.process = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 2, stride=2),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )
            self.final = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )   
        else:
            self.process = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 2, stride=2),
                nn.ReLU(inplace=True),
            )
            self.final = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, 3, padding = 1),
                nn.ReLU(inplace=True)
            )   

    def forward(self, x):
        xp = self.process(x)
        xi = F.interpolate(xp, scale_factor=0.5, mode='nearest')
        xc = self.downsample(xp)
        xr = self.final(torch.cat([xi, xc], 1))
        return xr

class UNetUpConvNearest(nn.Module):

    def __init__(self, in_channels, features, out_channels, is_bn = True):
        super().__init__()
        if is_bn:
            self.process = nn.Sequential(
                nn.Conv2d(in_channels, features, 3, padding = 1),
                nn.BatchNorm2d(features),
                nn.LeakyReLU(inplace=True),
            )
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(features, features, 2, stride=2),
                nn.BatchNorm2d(features),
                nn.LeakyReLU(inplace=True)
            )
            self.final1 = nn.Sequential(
                nn.Conv2d(features * 2, out_channels, 1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True)
            )
            self.final2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True)                
            )
        else:
            self.process = nn.Sequential(
                nn.Conv2d(in_channels, features, 3, padding = 1),
                nn.LeakyReLU(inplace=True),
            )
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(features, features, 2, stride=2),
                nn.LeakyReLU(inplace=True)
            )
            self.final1 = nn.Sequential(
                nn.Conv2d(features * 2, out_channels, 1, padding=0),
                nn.LeakyReLU(inplace=True)
            )
            self.final2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.LeakyReLU(inplace=True)                
            )

    def forward(self, x):
        xp = self.process(x)
        xi = F.interpolate(xp, scale_factor=2, mode='nearest')
        xc = self.upsample(xp)
        xr1 = self.final1(torch.cat([xi, xc], 1))
        xr2 = self.final2(xr1)      
        return xr1 + xr2
        
class UNetDownConvNearest(nn.Module):

    def __init__(self, in_channels, out_channels, is_bn = True):
        super().__init__()

        if is_bn:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 4, 2, 1),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(inplace=True),
            )
            self.process1 = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, 1, padding = 0),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True)
            )
            self.process2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True)
            )   
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 2, stride=2),
                nn.LeakyReLU(inplace=True),
            )
            self.process1 = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, 1, padding = 0),
                nn.LeakyReLU(inplace=True)
            )
            self.process2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.LeakyReLU(inplace=True)
            )
        

    def forward(self, x):
        xi = F.interpolate(x, scale_factor=0.5, mode='nearest')
        xc = self.downsample(x)
        xr1 = self.process1(torch.cat([xi, xc], 1))
        xr2 = self.process2(xr1)
        return xr1 + xr2



class DenseNetConnection(nn.Module):
    def __init__(self, channels, num_convs):
        super().__init__()
        self.channels = channels
        self.num_convs = num_convs
        self.layers = []
        self.first = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        for i in range(self.num_convs):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True))
            )
        self.layers = nn.ModuleList(self.layers)
    def forward(self, x):
        x_old = x
        x_new = self.first(x)
        x_list = [x_old.clone(), x_new.clone()]
        for i in range(self.num_convs):
            x_res = self.layers[i](sum(x_list))
            x_list.append(x_res.clone())
        return x_res

class DenseNetConnectionv2(nn.Module):
    def __init__(self, channels, num_convs):
        super().__init__()
        self.channels = channels
        self.num_convs = num_convs
        self.layers = []
        self.first = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True)
        )
        for i in range(self.num_convs):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(inplace=True))
            )
        self.layers = nn.ModuleList(self.layers)
    def forward(self, x):
        x_old = x
        x_new = self.first(x)
        x_list = [x_old.clone(), x_new.clone()]
        for i in range(self.num_convs):
            x_res = self.layers[i](sum(x_list))
            x_list.append(x_res.clone())
        return x_res

class ResNetConnection(nn.Module):
    def __init__(self, channels, num_convs):
        super().__init__()
        self.channels = channels
        self.num_convs = num_convs
        self.layers = []
        self.first = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        for i in range(self.num_convs):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, padding=0, stride=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        )
        self.layers = nn.ModuleList(self.layers)
    def forward(self, x):
        x_old = x
        x_new = self.first(x)
        for i in range(self.num_convs):
            x_res = self.layers[i](torch.cat([x_old, x_new], 1))
            x_old = x_new
            x_new = x_res
        return x_res
        
class DenseUNetPolar(nn.Module):

    def __init__(self, num_classes, num_channels, image_shape=(64,512), width=32):
        super().__init__()
        self.width = width
        self.first = nn.Sequential(
            nn.Conv2d(num_channels, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.dec2 = UNetDownConv(width, 2*width) 
        self.dec3 = UNetDownConv(2*width, 4*width) #120, 160 / 32, 256
        self.dec4 = UNetDownConv(4*width, 8*width)  #60, 80   / 16, 128

        self.center_downsample = nn.Sequential(
            nn.Conv2d(8*width, 8*width, 2, stride=2),
            nn.BatchNorm2d(8*width),
            nn.ReLU(inplace=True)
        )#30, 15 
        self.center1 = nn.Conv2d(16*width, 16*width, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(16*width)
        self.center_relu = nn.ReLU(inplace=True)    
        
        self.info_linear = nn.Sequential(
                              nn.Linear(2, 16*width),
                              nn.ReLU(inplace=True)
                          )
                
        self.center2 = nn.Conv2d(16*width, 8*width, 3, padding = 1, bias=False)
        self.center2_bn = nn.BatchNorm2d(8*width)
        self.center2_relu = nn.ReLU(inplace=True)
        
        self.center3 = nn.Conv2d(8*width, 8*width, 3, padding = 1)
        self.center3_bn = nn.BatchNorm2d(8*width)
        self.center3_relu = nn.ReLU(inplace=True) 
        
        self.center4 = nn.ConvTranspose2d(8*width, 8*width, 2, stride=2)
        self.center4_bn = nn.BatchNorm2d(8*width)
        self.center4_relu = nn.ReLU(inplace=True)
        self.center5 = nn.Conv2d(16*width, 8*width, 3, padding = 1)
        self.center5_bn = nn.BatchNorm2d(8*width)
        self.center5_relu = nn.ReLU(inplace=True)

        self.res4 = DenseNetConnection(channels=8*width, num_convs=2)
        self.res3 = DenseNetConnection(channels=4*width, num_convs=4)
        self.res2 = DenseNetConnection(channels=2*width, num_convs=6)
        self.res1 = DenseNetConnection(channels=width, num_convs=8)
        
        self.enc4 = UNetUpConv(16*width, 8*width, 4*width)
        self.enc3 = UNetUpConv(8*width, 4*width, 2*width)
        self.enc2 = UNetUpConv(4*width, 2*width, width)
        self.enc1 = nn.Sequential(
            nn.Conv2d(2*width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(width, num_classes, 1)
        )

    def forward(self, x, pr, ir):
        dec1 = self.first(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)

        center1_1 = self.center_downsample(dec4)
        center1_2 = F.interpolate(dec4, scale_factor=0.5, mode='bilinear', align_corners=True)
        center1 = torch.cat([center1_1, center1_2], 1)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        
        linear = self.info_linear(torch.cat([pr.view(-1,1), ir.view(-1,1)], 1))
        
        center5 = self.center2(center4+linear.view(-1, center4.shape[1], 1, 1))
        center6 = self.center2_bn(center5)
        center7 = self.center2_relu(center6)
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        center13i = F.interpolate(center10, scale_factor=2, mode='bilinear', align_corners=True)
        center11 = self.center4(center10)
        center12 = self.center4_bn(center11)
        center13 = self.center4_relu(center12)
        center14 = self.center5(torch.cat([center13, center13i], 1))
        center15 = self.center5_bn(center14)
        center16 = self.center5_relu(center15)

        enc4 = self.enc4(torch.cat([center16, self.res4(dec4)], 1))
        enc3 = self.enc3(torch.cat([enc4, self.res3(dec3)], 1))
        enc2 = self.enc2(torch.cat([enc3, self.res2(dec2)], 1))
        enc1 = self.enc1(torch.cat([enc2, self.res1(dec1)], 1))
        
        #enc4 = self.enc4(torch.cat([center16, dec4], 1))
        #enc3 = self.enc3(torch.cat([enc4, dec3], 1))
        #enc2 = self.enc2(torch.cat([enc3, dec2], 1))
        #enc1 = self.enc1(torch.cat([enc2, dec1], 1))

        return self.final(enc1)

class ResUNet(nn.Module):

    def __init__(self, num_classes, num_channels, width=32):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(num_channels, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.dec2 = UNetDownConv(width, 2*width) 
        self.dec3 = UNetDownConv(2*width, 4*width) #120, 160
        self.dec4 = UNetDownConv(4*width, 8*width)  #60, 80   

        self.center_downsample = nn.Sequential(
            nn.Conv2d(8*width, 8*width, 3, stride=2, padding = 1),
            nn.BatchNorm2d(8*width),
            nn.ReLU(inplace=True)
        )#30, 15 
        self.center1 = nn.Conv2d(16*width, 16*width, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(16*width)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(16*width, 16*width, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(16*width)
        self.center2_relu = nn.ReLU(inplace=True)
        self.center3 = nn.ConvTranspose2d(16*width, 8*width, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(8*width)
        self.center3_relu = nn.ReLU(inplace=True)

        self.res4 = ResNetConnection(channels=8*width, num_convs=2)
        self.res3 = ResNetConnection(channels=4*width, num_convs=4)
        self.res2 = ResNetConnection(channels=2*width, num_convs=6)
        self.res1 = ResNetConnection(channels=width, num_convs=8)
        
        self.enc4 = UNetUpConv(24*width, 8*width, 4*width)
        self.enc3 = UNetUpConv(12*width, 4*width, 2*width)
        self.enc2 = UNetUpConv(6*width, 2*width, width)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3*width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(width, num_classes, 1),
            nn.Tanh()
        )
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

        center1_1 = self.center_downsample(dec4)
        center1_2 = F.interpolate(dec4, scale_factor=0.5, mode='bilinear', align_corners=True)
        center1 = torch.cat([center1_1, center1_2], 1)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2relu(center6)
        
        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)

        enc4 = self.enc4(torch.cat([center10, dec4, self.res4(dec4)], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3, self.res3(dec3)], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2, self.res2(dec2)], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1, self.res1(dec1)], 1))

        return self.final(enc1)

class ResUNetv2(nn.Module):

    def __init__(self, num_classes, num_channels, width=32):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(num_channels, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.dec2 = UNetDownConv(width, 2*width) 
        self.dec3 = UNetDownConv(2*width, 4*width) #120, 160
        self.dec4 = UNetDownConv(4*width, 8*width)  #60, 80   

        self.center_downsample = nn.Sequential(
            nn.Conv2d(8*width, 8*width, 2, stride=2),
            nn.BatchNorm2d(8*width),
            nn.ReLU(inplace=True)
        )#30, 15 
        self.center1 = nn.Conv2d(16*width, 16*width, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(16*width)
        self.center_relu = nn.ReLU(inplace=True)
        self.center2 = nn.Conv2d(16*width, 16*width, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(16*width)
        self.center2_relu = nn.ReLU(inplace=True)
        
        self.center3i = nn.Conv2d(16*width, 8*width, 3, padding = 1)
        self.center3i_bn = nn.BatchNorm2d(8*width)
        self.center3i_relu = nn.ReLU(inplace=True)

        self.center3 = nn.ConvTranspose2d(16*width, 8*width, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(8*width)
        self.center3_relu = nn.ReLU(inplace=True)

        self.center4 = nn.Conv2d(16*width, 8*width, 3, padding = 1)
        self.center4_bn = nn.BatchNorm2d(8*width)
        self.center4_relu = nn.ReLU(inplace=True)

        self.res4 = ResNetConnection(channels=8*width, num_convs=2)
        self.res3 = ResNetConnection(channels=4*width, num_convs=4)
        self.res2 = ResNetConnection(channels=2*width, num_convs=6)
        self.res1 = ResNetConnection(channels=width, num_convs=8)
        
        self.enc4 = UNetUpConv(24*width, 8*width, 4*width)
        self.enc3 = UNetUpConv(12*width, 4*width, 2*width)
        self.enc2 = UNetUpConv(6*width, 2*width, width)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3*width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(width, num_classes, 1),
            nn.Tanh()
        )
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

        center1_1 = self.center_downsample(dec4)
        center1_2 = F.interpolate(dec4, scale_factor=0.5, mode='bilinear')
        center1 = torch.cat([center1_1, center1_2], 1)
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)

        center7 = self.center2_relu(center6)
        center7i = F.interpolate(center7, scale_factor=2, mode='bilinear')

        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)

        center8i = self.center3i(center7i)
        center9i = self.center3i_bn(center8i)
        center10i = self.center3i_relu(center9i)

        center11 = self.center4(torch.cat([center10, center10i], 1))
        center12 = self.center4_bn(center11)
        center13 = self.center4_relu(center12)

        enc4 = self.enc4(torch.cat([center13, dec4, self.res4(dec4)], 1))
        enc3 = self.enc3(torch.cat([enc4, dec3, self.res3(dec3)], 1))
        enc2 = self.enc2(torch.cat([enc3, dec2, self.res2(dec2)], 1))
        enc1 = self.enc1(torch.cat([enc2, dec1, self.res1(dec1)], 1))

        return self.final(enc1)
        
class MultiResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, res_list=[3,5,7]):
        super().__init__()
        
        self.convs = []
        
        for res in res_list:
            self.convs.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, res, padding = int((res-1)/2)),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True))
                )
        
        self.convs = nn.ModuleList(self.convs)
        
    def forward(self, x):
        conv_results = []
        #print('############################')
        for i in range(len(self.convs)):
            conv_result = self.convs[i](x)
            #print(conv_result.shape)
            conv_results.append(conv_result)
        #print('############################')
        return sum(conv_results)
    
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
    
class DenseNetConnectionMR(nn.Module):
    def __init__(self, channels, num_convs):
        super().__init__()
        self.channels = channels
        self.num_convs = num_convs
        self.layers = []
        self.first = MultiResConvBlock(channels, channels)
        for i in range(self.num_convs):
            self.layers.append(MultiResConvBlock(channels, channels))
        self.layers = nn.ModuleList(self.layers)
        
    def forward(self, x):
        x_old = x
        x_new = self.first(x)
        x_list = [x_old.clone(), x_new.clone()]
        for i in range(self.num_convs):
            x_res = self.layers[i](sum(x_list))
            x_list.append(x_res.clone())
        return x_res
    
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

class UNetUpMR(nn.Module):

    def __init__(self, in_channels, features, out_channels, is_bn = True):
        super().__init__()
        self.process = nn.Sequential(
                MultiResConvBlock(in_channels, features),
                MultiResConvBlock(features, out_channels)
            )
            
        self.upsample = nn.Sequential(
                nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
        self.final = MultiResConvBlock(out_channels * 2, out_channels)

    def forward(self, x):
        xp = self.process(x)
        xi = F.interpolate(xp, scale_factor=2, mode='bilinear', align_corners=True)
        xc = self.upsample(xp)
        xr = self.final(torch.cat([xi, xc], 1))        
        return xr
        
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

class UNetDownMR(nn.Module):

    def __init__(self, in_channels, out_channels, is_bn = True):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 2, stride=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.process = nn.Sequential(
            MultiResConvBlock(in_channels * 2, out_channels),
            MultiResConvBlock(out_channels, out_channels)
        )
        

    def forward(self, x):
        xi = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        xc = self.downsample(x)
        xr = self.process(torch.cat([xi, xc], 1))
        return xr
     
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

class MultiResDenseUNet(nn.Module):

    def __init__(self, num_classes, num_channels, width=4):
        super().__init__()
        
        self.first = MultiResConvBlock(num_channels, width)
        self.dec2 = UNetDownMR(width, 2*width) 
        self.dec3 = UNetDownMR(2*width, 4*width) #120, 160
        self.dec4 = UNetDownMR(4*width, 8*width)  #60, 80   

        self.center_downsample = nn.Sequential(
            nn.Conv2d(8*width, 8*width, 2, stride=2),
            nn.BatchNorm2d(8*width),
            nn.ReLU(inplace=True)
        )#30, 15 
                
        self.center1 = MultiResConvBlock(16*width, 16*width)
        self.center2 = MultiResConvBlock(32*width, 16*width)
        self.center3 = nn.Sequential(
                nn.ConvTranspose2d(16*width, 8*width, 2, stride=2),
                nn.BatchNorm2d(8*width),
                nn.ReLU(inplace=True)
        )
        self.center3i = MultiResConvBlock(16*width, 8*width)
        self.center4 = MultiResConvBlock(16*width, 8*width)

        self.dn4 = DenseNetConnectionMR(channels=8*width, num_convs=2)
        self.dn3 = DenseNetConnectionMR(channels=4*width, num_convs=4)
        self.dn2 = DenseNetConnectionMR(channels=2*width, num_convs=6)
        self.dn1 = DenseNetConnectionMR(channels=width, num_convs=8)
        
        self.enc4 = UNetUpMR(16*width, 8*width, 4*width)
        self.enc3 = UNetUpMR(8*width, 4*width, 2*width)
        self.enc2 = UNetUpMR(4*width, 2*width, width)
        self.enc1_1 = MultiResConvBlock(2*width, width)
        self.enc1_2 = MultiResConvBlock(2*width, width) 
        self.final = nn.Sequential(
            nn.Conv2d(width, num_classes, 1),
            nn.Tanh()
        )
        
        self._initialize_weights()
     
    def _initialize_weights(self):
        self.dec2._initialize_weights()
        self.dec3._initialize_weights()
        self.dec4._initialize_weights()
        self.center1._initialize_weights()
        self.center2._initialize_weights()
        self.center3i._initialize_weights()
        self.center4._initialize_weights()
        self.dn4._initialize_weights()
        self.dn3._initialize_weights()
        self.dn2._initialize_weights()
        self.dn1._initialize_weights()
        self.enc4._initialize_weights()
        self.enc3._initialize_weights()
        self.enc2._initialize_weights()
        self.enc1_1._initialize_weights()
        self.enc1_2._initialize_weights()
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

        center1_1 = self.center_downsample(dec4)
        center1_2 = F.interpolate(dec4, scale_factor=0.5, mode='bilinear', align_corners=True)
        center1 = torch.cat([center1_1, center1_2], 1)        
        center2 = self.center1(center1)
        center3 = self.center2(torch.cat([center2, center1], 1))
        center3i = F.interpolate(center3, scale_factor=2, mode='bilinear', align_corners=True)
        center4 = self.center3(center3)
        center4i = self.center3i(center3i)
        center5 = self.center4(torch.cat([center4, center4i], 1))
        enc4 = self.enc4(torch.cat([center5, self.dn4(dec4)], 1))
        enc3 = self.enc3(torch.cat([enc4, self.dn3(dec3)], 1))
        enc2 = self.enc2(torch.cat([enc3, self.dn2(dec2)], 1))
        enc1 = self.enc1_1(torch.cat([enc2, self.dn1(dec1)], 1))
        enc0 = self.enc1_2(torch.cat([enc1, enc2], 1))

        return self.final(enc0)

class UNetUpConvNearestv2(nn.Module):

    def __init__(self, in_channels, features, out_channels, is_bn = True):
        super().__init__()
        if is_bn:
            self.process = nn.Sequential(
                nn.Conv2d(in_channels, features, 3, padding = 1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            )
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(features, features, 2, stride=2),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            )
            self.final = nn.Sequential(
                nn.Conv2d(features * 2, features, 3, padding = 1),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, out_channels, 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.process = nn.Sequential(
                nn.Conv2d(in_channels, features, 3, padding = 1),
                nn.ReLU(inplace=True)
            )
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(features, features, 2, stride=2),
                nn.ReLU(inplace=True)
            )
            self.final = nn.Sequential(
                nn.Conv2d(features * 2, features, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, out_channels, 3, padding = 1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        xp = self.process(x)
        xi = F.interpolate(xp, scale_factor=2, mode='nearest')
        xc = self.upsample(xp)
        xr = self.final(torch.cat([xi, xc], 1))      
        return xr
        
class UNetDownConvNearestv2(nn.Module):

    def __init__(self, in_channels, out_channels, is_bn = True):
        
        super().__init__()
        if is_bn:
            self.first = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 2, stride=2),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
            self.process = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )   
        else:
            self.first = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 2, stride=2),
                nn.ReLU(inplace=True)
            )
            self.process = nn.Sequential(
                nn.Conv2d(in_channels * 2, out_channels, 3, padding = 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding = 1),
                nn.ReLU(inplace=True)
            )  
        

    def forward(self, x):
        xp = self.first(x)
        xi = F.interpolate(xp, scale_factor=0.5, mode='nearest')
        xc = self.downsample(xp)
        xr = self.process(torch.cat([xi, xc], 1))
        return xr

class DenseNetConnectionConv(nn.Module):
    def __init__(self, channels, num_convs):
        super().__init__()
        self.channels = channels
        self.num_convs = num_convs
        self.layers = []
        self.first = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        for i in range(self.num_convs):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels=channels*(i+2), out_channels=channels, kernel_size=1, padding=0, stride=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
                )
            )
        self.layers = nn.ModuleList(self.layers)
    def forward(self, x):
        x_old = x
        x_new = self.first(x)
        x_list = [x_old.clone(), x_new.clone()]
        for i in range(self.num_convs):
            x_res = self.layers[i](torch.cat(x_list, dim=1))
            x_list.append(x_res.clone())
        return x_res
      
class DenseUNetv2(nn.Module):

    def __init__(self, num_classes, num_channels, width=32):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(num_channels, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.dec2 = UNetDownConvNearestv2(width, 2*width) 
        self.dec3 = UNetDownConvNearestv2(2*width, 4*width) #120, 160
        self.dec4 = UNetDownConvNearestv2(4*width, 8*width)  #60, 80   

        self.center_downsample = nn.Sequential(
            nn.Conv2d(8*width, 8*width, 4, 2, 1),
            nn.BatchNorm2d(8*width),
            nn.ReLU(inplace=True)
        )#30, 15 
        self.center1 = nn.Conv2d(16*width, 16*width, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(16*width)
        self.center_relu = nn.ReLU(inplace=True)
        
        self.center2 = nn.Conv2d(16*width, 16*width, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(16*width)
        self.center2_relu = nn.ReLU(inplace=True)

        self.center3 = nn.ConvTranspose2d(16*width, 8*width, 4, 2, 1)
        self.center3_bn = nn.BatchNorm2d(8*width)
        self.center3_relu = nn.ReLU(inplace=True)
        
        self.center3i = nn.Conv2d(16*width, 8*width, 3, padding = 1)
        self.center3i_bn = nn.BatchNorm2d(8*width)
        self.center3i_relu = nn.ReLU(inplace=True)

        self.center4 = nn.Conv2d(16*width, 8*width, 3, padding = 1)
        self.center4_bn = nn.BatchNorm2d(8*width)
        self.center4_relu = nn.ReLU(inplace=True)        

        self.dn4 = DenseNetConnection(channels=8*width, num_convs=2)
        self.dn3 = DenseNetConnection(channels=4*width, num_convs=4)
        self.dn2 = DenseNetConnection(channels=2*width, num_convs=8)
        self.dn1 = DenseNetConnection(channels=width, num_convs=16)
        
        self.enc4 = UNetUpConvNearestv2(16*width, 8*width, 4*width)
        self.enc3 = UNetUpConvNearestv2(8*width, 4*width, 2*width)
        self.enc2 = UNetUpConvNearestv2(4*width, 2*width, width)
        self.enc1 = nn.Sequential(
            nn.Conv2d(2*width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(width, num_classes, 1, padding=0),
            nn.Tanh()
        )
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

        center1_1 = self.center_downsample(dec4)
        center1_2 = F.interpolate(dec4, scale_factor=0.5, mode='nearest')
        center1 = torch.cat([center1_1, center1_2], 1)
        
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2_relu(center6)
        center7i = F.interpolate(center7, scale_factor=2, mode='nearest')

        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        center8i = self.center3i(center7i)
        center9i = self.center3i_bn(center8i)
        center10i = self.center3i_relu(center9i)

        center11 = self.center4(torch.cat([center10, center10i], 1))
        center12 = self.center4_bn(center11)
        center13 = self.center4_relu(center12)

        enc4 = self.enc4(torch.cat([center13, self.dn4(dec4)], 1))
        enc3 = self.enc3(torch.cat([enc4, self.dn3(dec3)], 1))
        enc2 = self.enc2(torch.cat([enc3, self.dn2(dec2)], 1))
        enc1 = self.enc1(torch.cat([enc2, self.dn1(dec1)], 1))

        return self.final(enc1)

class DenseUNet(nn.Module):

    def __init__(self, num_classes, num_channels, width=32):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(num_channels, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.dec2 = UNetDownConv(width, 2*width) 
        self.dec3 = UNetDownConv(2*width, 4*width) #120, 160
        self.dec4 = UNetDownConv(4*width, 8*width)  #60, 80   

        self.center_downsample = nn.Sequential(
            nn.Conv2d(8*width, 8*width, 2, stride=2),
            nn.BatchNorm2d(8*width),
            nn.ReLU(inplace=True)
        )#30, 15 
        self.center1 = nn.Conv2d(16*width, 16*width, 3, padding = 1)
        self.center_bn = nn.BatchNorm2d(16*width)
        self.center_relu = nn.ReLU(inplace=True)
        
        self.center2 = nn.Conv2d(16*width, 16*width, 3, padding = 1)
        self.center2_bn = nn.BatchNorm2d(16*width)
        self.center2_relu = nn.ReLU(inplace=True)

        self.center3 = nn.ConvTranspose2d(16*width, 8*width, 2, stride=2)
        self.center3_bn = nn.BatchNorm2d(8*width)
        self.center3_relu = nn.ReLU(inplace=True)
        
        self.center3i = nn.Conv2d(16*width, 8*width, 3, padding = 1)
        self.center3i_bn = nn.BatchNorm2d(8*width)
        self.center3i_relu = nn.ReLU(inplace=True)
        
        self.center4 = nn.Conv2d(16*width, 8*width, 3, padding = 1)
        self.center4_bn = nn.BatchNorm2d(8*width)
        self.center4_relu = nn.ReLU(inplace=True)

        self.dn4 = DenseNetConnection(channels=8*width, num_convs=2)
        self.dn3 = DenseNetConnection(channels=4*width, num_convs=4)
        self.dn2 = DenseNetConnection(channels=2*width, num_convs=6)
        self.dn1 = DenseNetConnection(channels=width, num_convs=8)
        
        self.enc4 = UNetUpConv(16*width, 8*width, 4*width)
        self.enc3 = UNetUpConv(8*width, 4*width, 2*width)
        self.enc2 = UNetUpConv(4*width, 2*width, width)
        self.enc1 = nn.Sequential(
            nn.Conv2d(2*width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding = 1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(width, num_classes, 1),
            nn.Tanh()
        )
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

        center1_1 = self.center_downsample(dec4)
        center1_2 = F.interpolate(dec4, scale_factor=0.5, mode='nearest')
        center1 = torch.cat([center1_1, center1_2], 1)
        
        center2 = self.center1(center1)
        center3 = self.center_bn(center2)
        center4 = self.center_relu(center3)
        
        center5 = self.center2(center4)
        center6 = self.center2_bn(center5)
        center7 = self.center2_relu(center6)
        center7i = F.interpolate(center7, scale_factor=2, mode='nearest')

        center8 = self.center3(center7)
        center9 = self.center3_bn(center8)
        center10 = self.center3_relu(center9)
        
        center8i = self.center3i(center7i)
        center9i = self.center3i_bn(center8i)
        center10i = self.center3i_relu(center9i)
        
        center11 = self.center4(torch.cat([center10, center10i], 1))
        center12 = self.center4_bn(center11)
        center13 = self.center4_relu(center12)

        enc4 = self.enc4(torch.cat([center13, self.dn4(dec4)], 1))
        enc3 = self.enc3(torch.cat([enc4, self.dn3(dec3)], 1))
        enc2 = self.enc2(torch.cat([enc3, self.dn2(dec2)], 1))
        enc1 = self.enc1(torch.cat([enc2, self.dn1(dec1)], 1))

        return self.final(enc1)

class TVLoss(nn.Module):
    """Total variation loss (Lp penalty on image gradient magnitude).
    The input must be 4D. If a target (second parameter) is passed in, it is
    ignored.
    ``p=1`` yields the vectorial total variation norm. It is a generalization
    of the originally proposed (isotropic) 2D total variation norm (see
    (see https://en.wikipedia.org/wiki/Total_variation_denoising) for color
    images. On images with a single channel it is equal to the 2D TV norm.
    ``p=2`` yields a variant that is often used for smoothing out noise in
    reconstructions of images from neural network feature maps (see Mahendran
    and Vevaldi, "Understanding Deep Image Representations by Inverting
    Them", https://arxiv.org/abs/1412.0035)
    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.
    """

    def __init__(self, p, reduction='mean', eps=1e-8):
        super().__init__()
        if p not in {1, 2}:
            raise ValueError('p must be 1 or 2')
        if reduction not in {'mean', 'sum', 'none'}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.p = p
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target=None):
        input = F.pad(input, (0, 1, 0, 1), 'replicate')
        x_diff = input[..., :-1, :-1] - input[..., :-1, 1:]
        y_diff = input[..., :-1, :-1] - input[..., 1:, :-1]
        diff = x_diff**2 + y_diff**2
        if self.p == 1:
            diff = (diff + self.eps).mean(dim=1, keepdims=True).sqrt()
        if self.reduction == 'mean':
            return diff.mean()
        if self.reduction == 'sum':
            return diff.sum()
        return diff

       
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, device, size=(168, 224), resize=False, model='vgg19'):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        if model == 'vgg19':
            blocks.append(models.vgg19(pretrained=True).features[:4].eval().to(device))
            blocks.append(models.vgg19(pretrained=True).features[4:9].eval().to(device))
            blocks.append(models.vgg19(pretrained=True).features[9:18].eval().to(device))
            blocks.append(models.vgg19(pretrained=True).features[18:27].eval().to(device))
        elif model == 'vgg16':
            blocks.append(models.vgg19(pretrained=True).features[:4].eval().to(device))
            blocks.append(models.vgg19(pretrained=True).features[4:9].eval().to(device))
            blocks.append(models.vgg19(pretrained=True).features[9:16].eval().to(device))
            blocks.append(models.vgg19(pretrained=True).features[16:23].eval().to(device))
        self.blocks = torch.nn.ModuleList(blocks)
        self.resize = resize
        self.size = size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=None):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = F.interpolate(input, mode='bilinear', size=self.size, align_corners=True)
            target = F.interpolate(target, mode='bilinear', size=self.size, align_corners=True)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += F.l1_loss(x, y)
            if style_layers is not None:
                if i in style_layers:
                    act_x = x.reshape(x.shape[0], x.shape[1], -1)
                    act_y = y.reshape(y.shape[0], y.shape[1], -1)
                    gram_x = act_x @ act_x.permute(0, 2, 1)
                    gram_y = act_y @ act_y.permute(0, 2, 1)
                    loss += F.l1_loss(gram_x, gram_y)
        return loss  

class LPIPSLoss(torch.nn.Module):
    def __init__(self, device):
        super(LPIPSLoss, self).__init__()
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        return self.lpips_loss(input, target)

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
        
class UpsampleN(nn.Module):

    def __init__(self, channels, is_bn = True):
        super().__init__()
        if is_bn:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(channels, channels, 2, stride=2),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            self.final = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(channels, channels, 2, stride=2),
                nn.ReLU(inplace=True)
            )
            self.final = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        xi = F.interpolate(x, scale_factor=2, mode='nearest')
        xc = self.upsample(x)
        xr = self.final(torch.cat([xi, xc], 1))        
        return xr

class Upsample3(nn.Module):

    def __init__(self, channels, is_bn = True):
        super().__init__()
        if is_bn:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(channels, channels, 2, stride=2),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            self.final = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, padding=0),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(channels, channels, 2, stride=2),
                nn.ReLU(inplace=True)
            )
            self.final = nn.Sequential(
                nn.Conv2d(channels * 3, channels, 1, padding=0),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        xi = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        xn = F.interpolate(x, scale_factor=2, mode='nearest')
        xc = self.upsample(x)
        xr = self.final(torch.cat([xi, xn, xc], 1))        
        return xr

class UpsampleNearest(nn.Module):

    def __init__(self, channels, is_bn = True):
        super().__init__()
            
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        return x

class DownsampleN(nn.Module):

    def __init__(self, channels, is_bn = True):
        super().__init__()

        if is_bn:
            self.downsample = nn.Sequential(
                nn.Conv2d(channels, channels, 2, stride=2),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.process = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 3, padding = 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )   
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(channels, channels, 2, stride=2),
                nn.ReLU(inplace=True),
            )
            self.process = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 3, padding = 1),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
        xi = F.interpolate(x, scale_factor=0.5, mode='nearest')
        xc = self.downsample(x)
        xr = self.process(torch.cat([xi, xc], 1))
        return xr

class DownsampleNearest(nn.Module):

    def __init__(self, channels, is_bn = True):
        super().__init__()
    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='nearest')
        return x
        
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class NestedUNet(nn.Module):
    def __init__(self, num_classes, num_channels, width=32):
        super().__init__()

        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.conv0_0 = VGGBlock(num_channels, nb_filter[0], nb_filter[0])
        self.pool0_0 = DownsampleNearest(nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.up1_0 = UpsampleNearest(nb_filter[1])
        self.pool1_0 = DownsampleNearest(nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.up2_0 = UpsampleNearest(nb_filter[2])
        self.pool2_0 = DownsampleNearest(nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.up3_0 = UpsampleNearest(nb_filter[3])
        self.pool3_0 = DownsampleNearest(nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.up4_0 = UpsampleNearest(nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1_1 = UpsampleNearest(nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.up2_1 = UpsampleNearest(nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.up3_1 = UpsampleNearest(nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1_2 = UpsampleNearest(nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        self.up2_2 = UpsampleNearest(nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1_3 = UpsampleNearest(nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        
        self.final = nn.Conv2d(nb_filter[0]*4, num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool0_0(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool1_0(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool2_0(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool3_0(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1))
        
        output = self.final(torch.cat([x0_1, x0_2, x0_3, x0_4], 1))
            
        return nn.Tanh()(output)


class VGGBlockv2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class Downsamplev2(nn.Module):

    def __init__(self, channels, is_bn = True):
        super().__init__()

        if is_bn:
            self.process1 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding = 1),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )   
            self.process2 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding = 1),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.process1 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding = 1),
                nn.LeakyReLU(0.2, inplace=True)
            )   
            self.process2 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding = 1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
    def forward(self, x):
        x1 = self.process1(x)
        x2 = self.process2(x1)
        x3 = F.interpolate(x + x1 + x2, scale_factor=0.5, mode='nearest')
        return x3

class Upsamplev2(nn.Module):

    def __init__(self, channels, is_bn = True):
        super().__init__()

        if is_bn:
            self.process1 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding = 1),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )   
            self.process2 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding = 1),
                nn.BatchNorm2d(channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.process1 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding = 1),
                nn.LeakyReLU(0.2, inplace=True)
            )   
            self.process2 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding = 1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
    def forward(self, x):
        x1 = F.interpolate(x, scale_factor=2, mode='nearest')
        x2 = self.process1(x1)
        x3 = self.process2(x2)
        return x1 + x2 + x3

class NestedUNetv2(nn.Module):
    def __init__(self, num_classes, num_channels, width=32, deep_supervision=False):
        super().__init__()

        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.deep_supervision = deep_supervision

        self.conv0_0 = VGGBlockv2(num_channels, nb_filter[0], nb_filter[0])
        self.pool0_0 = Downsamplev2(nb_filter[0])
        self.conv1_0 = VGGBlockv2(nb_filter[0], nb_filter[1], nb_filter[1])
        self.up1_0 = Upsamplev2(nb_filter[1])
        self.pool1_0 = Downsamplev2(nb_filter[1])
        self.conv2_0 = VGGBlockv2(nb_filter[1], nb_filter[2], nb_filter[2])
        self.up2_0 = Upsamplev2(nb_filter[2])
        self.pool2_0 = Downsamplev2(nb_filter[2])
        self.conv3_0 = VGGBlockv2(nb_filter[2], nb_filter[3], nb_filter[3])
        self.up3_0 = Upsamplev2(nb_filter[3])
        self.pool3_0 = Downsamplev2(nb_filter[3])
        self.conv4_0 = VGGBlockv2(nb_filter[3], nb_filter[4], nb_filter[4])
        self.up4_0 = Upsamplev2(nb_filter[4])

        self.conv0_1 = VGGBlockv2(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlockv2(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1_1 = Upsamplev2(nb_filter[1])
        self.conv2_1 = VGGBlockv2(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.up2_1 = Upsamplev2(nb_filter[2])
        self.conv3_1 = VGGBlockv2(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.up3_1 = Upsamplev2(nb_filter[3])

        self.conv0_2 = VGGBlockv2(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlockv2(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1_2 = Upsamplev2(nb_filter[1])
        self.conv2_2 = VGGBlockv2(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        self.up2_2 = Upsamplev2(nb_filter[2])

        self.conv0_3 = VGGBlockv2(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlockv2(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.up1_3 = Upsamplev2(nb_filter[1])

        self.conv0_4 = VGGBlockv2(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool0_0(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool1_0(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool2_0(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool3_0(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class SIFLayerMask(nn.Module):
    def __init__(self, polar_height, polar_width, filter_mat, device, channels = 1):
        super().__init__()
        self.device = device
        self.polar_height = polar_height
        self.polar_width = polar_width
        self.channels = channels
        self.angles = np.arange(0, 2 * np.pi, 2 * np.pi / self.polar_width)
        self.cos_angles = np.zeros((self.polar_width))
        self.sin_angles = np.zeros((self.polar_width))
        for i in range(self.polar_width):
            self.cos_angles[i] = np.cos(self.angles[i])
            self.sin_angles[i] = np.sin(self.angles[i])
        assert filter_mat.shape[0] == filter_mat.shape[1]
        self.filter_size = filter_mat.shape[0]
        self.num_filters = filter_mat.shape[2]
        self.filter = torch.FloatTensor(filter_mat).to(self.device).requires_grad_(False)
        self.filter_mat = filter_mat
        self.filter = torch.moveaxis(self.filter.unsqueeze(0), 3, 0)
        self.filter = torch.flip(self.filter, dims=[0]).requires_grad_(False)
        
    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
    
        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    
        G = torch.mm(features, features.t())  # compute the gram product
    
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)
       
    def grid_sample(self, input, grid, interp_mode='bilinear'):

        # grid: [-1, 1]
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1).to(self.device)
        return torch.nn.functional.grid_sample(input, newgrid, mode=interp_mode)

    def cartToPol(self, image, mask, pupil_xyr, iris_xyr):
        
        image_scaled = (image * 255).requires_grad_(True)

        batch_size = image_scaled.shape[0]
        width = image_scaled.shape[3]
        height = image_scaled.shape[2]

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

        image_polar = self.grid_sample(image_scaled, grid_sample_mat, interp_mode='bilinear')
        if mask is not None:
            mask_t = torch.tensor(mask).float()
            mask_t /= mask_t.max()
            one_mask = torch.ones(mask.shape).to(self.device)
            zero_mask = torch.zeros(mask.shape).to(self.device)
            mask_t = torch.where(mask_t > 0.5, one_mask, zero_mask)
            mask_polar = self.grid_sample(mask_t, grid_sample_mat, interp_mode='nearest')
        else:
            mask_polar = None

        return image_polar, mask_polar

    def getCodes(self, image_polar):
        r = int(np.floor(self.filter_size / 2))
        imgWrap = torch.zeros((image_polar.shape[0], image_polar.shape[1], r*2+self.polar_height, r*2+self.polar_width)).requires_grad_(False).to(self.device)
        
        imgWrap[:, :, :r, :r] += torch.clone(image_polar[:, :, -r:, -r:]).requires_grad_(True)
        imgWrap[:, :, :r, r:-r] += torch.clone(image_polar[:, :, -r:, :]).requires_grad_(True)
        imgWrap[:, :, :r, -r:] += torch.clone(image_polar[:, :, -r:, :r]).requires_grad_(True)

        imgWrap[:, :, r:-r, :r] += torch.clone(image_polar[:, :, :, -r:]).requires_grad_(True)
        imgWrap[:, :, r:-r, r:-r] += torch.clone(image_polar).requires_grad_(True)
        imgWrap[:, :, r:-r, -r:] += torch.clone(image_polar[:, :, :, :r]).requires_grad_(True)

        imgWrap[:, :, -r:, :r] += torch.clone(image_polar[:, :, :r, -r:]).requires_grad_(True)
        imgWrap[:, :, -r:, r:-r] += torch.clone(image_polar[:, :, :r, :]).requires_grad_(True)
        imgWrap[:, :, -r:, -r:] += torch.clone(image_polar[:, :, :r, :r]).requires_grad_(True)
        
        codes = nn.functional.conv2d(imgWrap, self.filter, stride=1, padding='valid')

        return codes

    def forward(self, image, pupil_xyr, iris_xyr, mask=None):
        image_polar, mask_polar = self.cartToPol(image, mask, pupil_xyr, iris_xyr)
        codes = self.getCodes(image_polar)
        codes_gram = self.gram_matrix(codes)  
        return codes, codes_gram, image_polar, mask_polar
        
class SIFLayerPolar(nn.Module):
    def __init__(self, filter_mat, device):
        super().__init__()
        self.device = device
        self.filter_size = filter_mat.shape[0]
        self.num_filters = filter_mat.shape[2]
        self.filter = torch.FloatTensor(filter_mat).to(self.device).requires_grad_(False)
        self.filter_mat = filter_mat
        self.filter = torch.moveaxis(self.filter.unsqueeze(0), 3, 0)
        self.filter = torch.flip(self.filter, dims=[0]).requires_grad_(False)

    def getCodes(self, image_polar):
        r = int(np.floor(self.filter_size / 2))
        imgWrap = torch.zeros((image_polar.shape[0], image_polar.shape[1], r*2+image_polar.shape[2], r*2+image_polar.shape[3])).requires_grad_(False).to(self.device)
        
        imgWrap[:, :, :r, :r] += torch.clone(image_polar[:, :, -r:, -r:]).requires_grad_(True)
        imgWrap[:, :, :r, r:-r] += torch.clone(image_polar[:, :, -r:, :]).requires_grad_(True)
        imgWrap[:, :, :r, -r:] += torch.clone(image_polar[:, :, -r:, :r]).requires_grad_(True)

        imgWrap[:, :, r:-r, :r] += torch.clone(image_polar[:, :, :, -r:]).requires_grad_(True)
        imgWrap[:, :, r:-r, r:-r] += torch.clone(image_polar).requires_grad_(True)
        imgWrap[:, :, r:-r, -r:] += torch.clone(image_polar[:, :, :, :r]).requires_grad_(True)

        imgWrap[:, :, -r:, :r] += torch.clone(image_polar[:, :, :r, -r:]).requires_grad_(True)
        imgWrap[:, :, -r:, r:-r] += torch.clone(image_polar[:, :, :r, :]).requires_grad_(True)
        imgWrap[:, :, -r:, -r:] += torch.clone(image_polar[:, :, :r, :r]).requires_grad_(True)
        
        codes = nn.functional.conv2d(imgWrap, self.filter, stride=1, padding='valid')

        return codes

    def forward(self, image_polar):
        codes = self.getCodes(image_polar) 
        return codes

'''
def matchCodes(self, code1, code2, mask1, mask2):
        
        if code1 is None or mask1 is None:
            return -1., 0.
        if code2 is None or mask2 is None:
            return -2., 0.

        margin = int(np.ceil(self.filter_size/2))
        code1 = np.array(code1)
        code2 = np.array(code2)
        mask1 = np.array(mask1)
        mask2 = np.array(mask2)
        
        self.code1 = code1[margin:-margin, :, :]
        self.code2 = code2[margin:-margin, :, :]
        self.mask1 = mask1[margin:-margin, :]
        self.mask2 = mask2[margin:-margin, :]

        scoreC = np.zeros((self.num_filters, 2*self.max_shift+1))
        for shift in range(-self.max_shift, self.max_shift+1):
            andMasks = np.logical_and(self.mask1, np.roll(self.mask2, shift, axis=1))
            xorCodes = np.logical_xor(self.code1, np.roll(self.code2, shift, axis=1))
            xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=2),self.num_filters))
            scoreC[:,shift] = np.sum(xorCodesMasked, axis=(0,1)) / np.sum(andMasks)

        scoreMean = np.mean(scoreC, axis=0)
        scoreC = np.min(scoreMean)
        scoreC_shift = self.max_shift-np.argmin(scoreMean)

        return scoreC, scoreC_shift
'''
'''
class ShiftedSIFLoss(nn.Module):
    def __init__(self, polar_height, polar_width, filter_mat, max_shift, device, sif_type = 'hinge', use_gram=False, channels = 1):
        super().__init__()
        self.max_shift = max_shift
        self.device = device
        self.polar_height = polar_height
        self.polar_width = polar_width
        self.filter_size = filter_mat.shape[0]
        self.sif_type = sif_type
        self.margin = int(np.ceil(self.filter_size/2))
        self.use_gram = use_gram
        self.sifLayerMask = SIFLayerMask(polar_height = polar_height, polar_width = polar_width, filter_mat = filter_mat, channels=channels, device=device).to(device)
    def forward(self, input, target, mask, pxyr, ixyr):
        sif_out, sif_out_gram, out_img_polar, out_mask_polar = self.sifLayerMask(input, pxyr, ixyr, mask=mask)
        sif_tar, sif_tar_gram, tar_img_polar, tar_mask_polar = self.sifLayerMask(target, pxyr, ixyr, mask=mask)
        sif_tar_ng = sif_tar.clone().detach().requires_grad_(False)
        one = torch.ones(sif_tar.shape).to(self.device)
        zero = torch.zeros(sif_tar_ng.shape).to(self.device)
        sif_tar_binary = torch.where(sif_tar_ng > 0, one, zero).requires_grad_(False)
        sif_out_binary = torch.where(sif_out > 0, one, zero).requires_grad_(False)
        tar_mask_polar_rep = torch.cat([tar_mask_polar]*7, dim=1).requires_grad_(False)
        
        for b in sif_out_binary.shape[0]:
            code1 = sif_out_binary[b, :, self.margin:-self.margin, :]
            code2 = sif_tar_binary[b, :, self.margin:-self.margin, :]
            mask1 = tar_mask_polar_rep[b, :, self.margin:-self.margin, :]
            mask2 = tar_mask_polar_rep[b, :, self.margin:-self.margin, :]
            scoreC = np.zeros((self.num_filters, 2*self.max_shift+1))
            for shift in range(-self.max_shift, self.max_shift+1):
                andMasks = np.logical_and(self.mask1, np.roll(self.mask2, shift, axis=1))
                xorCodes = np.logical_xor(self.code1, np.roll(self.code2, shift, axis=1))
                xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=2),self.num_filters))
                scoreC[:,shift] = np.sum(xorCodesMasked, axis=(0,1)) / np.sum(andMasks)
            scoreMean = np.mean(scoreC, axis=0)
            scoreC = np.min(scoreMean)
            scoreC_shift = self.max_shift-np.argmin(scoreMean)

        sif_shape = sif_out.shape
        sif_out_masked = (nn.Sigmoid()(sif_out) * tar_mask_polar_rep).requires_grad_(True)
        sif_tar_binary_masked = torch.clamp((sif_tar_binary * tar_mask_polar_rep).requires_grad_(False), 0, 1).float()
        sif_tar_masked = (nn.Sigmoid()(sif_tar_ng) * tar_mask_polar_rep).requires_grad_(False)
        minus_one = -1 * torch.ones(sif_tar.shape).to(self.device)
        sif_tar_hinge_masked = torch.where(sif_tar_binary_masked < 0.5, one, minus_one)
        sif_out_hinge_masked = (nn.Tanh()(sif_out) * tar_mask_polar_rep).requires_grad_(True)
        sif_out_hinge_masked = torch.where(sif_out_hinge_masked == 0, minus_one, sif_out_hinge_masked)
        if self.sif_type == 'bce':
            sif_out_sigmoid = nn.Sigmoid()(sif_out)
            #sif_tar_bce_masked = torch.where(tar_mask_polar_rep == 0, sif_out_sigmoid, nn.Sigmoid()(sif_tar_ng))
            sif_tar_bce_masked = torch.where(tar_mask_polar_rep == 0, sif_out_sigmoid, sif_tar_binary)
            sif_loss = nn.BCEWithLogitsLoss()(sif_out, sif_tar_bce_masked)
            #sif_loss = nn.BCELoss(reduction='sum')(sif_out_masked, sif_tar_masked) / torch.sum(tar_mask_polar_rep)   
        elif self.sif_type == 'hinge':
            sif_loss = nn.HingeEmbeddingLoss()(nn.Flatten()(sif_out_hinge_masked), nn.Flatten()(sif_tar_hinge_masked)) + 1
        else:
            #sif_loss = L1LossWithSoftLabels()(sif_out * tar_mask_polar_rep.float(), sif_tar_ng * tar_mask_polar_rep.float())
            sif_tar_l1_masked = torch.where(sif_tar_binary_masked > 0.5, one, minus_one)
            sif_loss = L1LossWithSoftLabels(device)(sif_out_hinge_masked, sif_tar_l1_masked)
        if self.use_gram:
            sif_loss += nn.L1Loss()(sif_out_gram, sif_tar_gram)
        
        return sif_loss
'''

class SoftMarginLossWithSoftLabels(nn.Module):
    def __init__(self, device, label_smoothing=0.2):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.device = device
    def forward(self, input, target):
        zero = torch.zeros(target.shape).to(self.device)
        noise_neg = torch.rand(input.shape).to(self.device)*self.label_smoothing
        noise_neg = torch.where(target < 0, noise_neg, zero)
        noise_pos = torch.rand(input.shape).to(self.device)*self.label_smoothing
        noise_pos = torch.where(target > 0, noise_pos, zero)
        target_soft = target + noise_neg - noise_pos
        return nn.SoftMarginLoss()(input, target_soft.requires_grad_(False))

class L1LossWithSoftLabels(nn.Module):
    def __init__(self, device, label_smoothing=0.2):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.device = device
    def forward(self, input, target):
        zero = torch.zeros(target.shape).to(self.device)
        targetm1 = torch.clamp(target + torch.rand(target.shape).to(self.device)*self.label_smoothing, -1 ,1)
        target1 = torch.clamp(target - torch.rand(target.shape).to(self.device)*self.label_smoothing, -1, 1)
        target_soft = torch.where(target == -1, targetm1, zero) + torch.where(target == 1, target1, zero)
        return nn.L1Loss()(input, target_soft)

class BCELogitsLossWithSoftLabels(nn.Module):
    def __init__(self, device, label_smoothing=0.2):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.device = device
    def forward(self, input, target, mask):
        zero = torch.zeros(target.shape).to(self.device)
        target0 = torch.clamp(target + torch.rand(target.shape).to(self.device)*self.label_smoothing, 0 ,1)
        target1 = torch.clamp(target - torch.rand(target.shape).to(self.device)*self.label_smoothing, 0, 1)
        target_soft = torch.where(target == 0, target0, zero) + torch.where(target == 1, target1, zero)
        return nn.BCEWithLogitsLoss(pos_weight=mask)(input, target_soft)