import numpy as np
import torch
import os
import math

import cv2
from argparse import ArgumentParser
import torch.nn as nn


from torch.optim import Adam, AdamW, SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torch.nn.functional import interpolate

from dataset import PairFromBinDatasetSB, PairMinMaxBinDataset
from network import UNet, SIFLayerMask, TVLoss, VGGPerceptualLoss, DenseUNet, LPIPSLoss, L1LossWithSoftLabels, HingeLossWithSoftLabels, Discriminator, DiscPerceptLoss, DenseUNetv2, NestedUNet, NestedUNetv2
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

if __name__ == '__main__':
    #print('Its running')
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--parent_dir', type=str, default='/data1/warsaw_pupil_dynamics')
    parser.add_argument('--train_bins_path', type=str, default='/data1/warsaw_pupil_dynamics/train_bins.pkl')
    parser.add_argument('--val_bins_path', type=str, default='/data1/warsaw_pupil_dynamics/val_bins.pkl')
    parser.add_argument('--test_bins_path', type=str, default='/data1/warsaw_pupil_dynamics/test_bins.pkl')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--use_mask_loss', action='store_true')
    parser.add_argument('--mask_model_path', type=str, default='./models/CCNet_epoch_260_NIRRGBmixed_adam.pth')
    parser.add_argument('--use_sif_loss', action='store_true')
    parser.add_argument('--sif_filter_path', type=str, default='./models/ICAtextureFilters_15x15_7bit.mat')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=20.0)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--log_batch', type=int, default=10)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--evaluate_minmax', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--network_pkl', type=str, default='./models/network-snapshot-iris.pkl')
    parser.add_argument('--val_repeats', type=int, default=10)
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--cfg_path', type=str, default="cfg.yaml", help="path of the iris recognition module configuration file.")
    parser.add_argument('--optim_type', type=str, default='adam')
    parser.add_argument('--virtual_batch_mult', type=int, default=4)
    parser.add_argument('--sif_bce', action='store_true')
    parser.add_argument('--use_tv_loss', action='store_true')
    parser.add_argument('--use_vgg_loss', action='store_true')
    parser.add_argument('--use_lpips_loss', action='store_true')
    parser.add_argument('--sif_hinge', action='store_true')
    parser.add_argument('--make_video', action='store_true')
    parser.add_argument('--vid_src', type=str, default='/data1/warsaw_pupil_dynamics/WarsawData/')
    parser.add_argument('--disc_resize', action='store_true')
    parser.add_argument('--use_disc_loss', action='store_true')
    parser.add_argument('--res_mult', type=int, default=1)
    parser.add_argument('--model_type', type=str, default='nestedunet')
    parser.add_argument('--sif_label_smoothing', action='store_true')
    parser.add_argument('--mask_bce', action='store_true')
    
    args = parser.parse_args()

    print('Test:', len(PairMinMaxBinDataset(args.test_bins_path, args.parent_dir).bins.keys()))
    print('Val:', len(PairMinMaxBinDataset(args.val_bins_path, args.parent_dir).bins.keys()))
    print('Train:', len(PairMinMaxBinDataset(args.train_bins_path, args.parent_dir).bins.keys()))
