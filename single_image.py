import numpy as np
import torch
import os
import math

import cv2
from argparse import ArgumentParser
import torch.nn as nn


from torch.optim import Adam, AdamW, SGD, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.functional import interpolate

from dataset import PairFromBinDatasetSB, PairMinMaxBinDataset
from network import UNet, SIFLayerMask, TVLoss, VGGPerceptualLoss, DenseUNet, LPIPSLoss, L1LossWithSoftLabels, HingeLossWithSoftLabels, DenseUNetv2, NestedUNet, NestedUNetv2, HingeLoss
from scipy import io
import datetime
import os
from typing import List, Optional, Tuple, Union
from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg

from torchvision import models, transforms
from skimage import img_as_bool

from tqdm import tqdm


from math import pi

from PIL import Image, ImageDraw

from scipy import io
#from modules.layers import ConvOffset2D

import math
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.deterministic = True

def img_transform(img_tensor):
    img_t = img_tensor[0]
    img_t = np.clip(img_t.clone().detach().cpu().numpy() * 255, 0, 255)
    img = Image.fromarray(img_t.astype(np.uint8))
    return img



device = torch.device('cpu')

deform_net = torch.load(sys.argv[1], map_location=device).to(device)
deform_net.eval()

pupil_img_path = sys.argv[2]
mask_img_path = sys.argv[3]

s_img = Image.open(pupil_img_path).convert('L')
b_mask = Image.open(mask_img_path).convert('L')
    
tensor_transform = transforms.Compose([
        transforms.Resize(size=(240, 320)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
])

s_img_t = tensor_transform(s_img).unsqueeze(0)
b_mask_t = tensor_transform(b_mask).unsqueeze(0)

inp = Variable(torch.cat([s_img_t, b_mask_t], dim=1)).to(device) 

out = deform_net(inp)

out_im = img_transform((out[0]+1)/2)
out_im.save('dilated.png')

