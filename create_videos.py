import numpy as np
import torch
import os
import math

import cv2
from argparse import ArgumentParser
import torch.nn as nn

from torch.optim import Adam, SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torch.nn.functional import interpolate

from dataset import PairFromBinDatasetSB
from network import UNet, SIFLayerMask, TVLoss, VGGPerceptualLoss, DenseUNet, LPIPSLoss, L1LossWithSoftLabels, HingeLossWithSoftLabels
from scipy import io
import datetime
import os
from typing import List, Optional, Tuple, Union
from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg

from torchvision import models, transforms
from skimage import img_as_bool

from tqdm import tqdm

from torchvision import models, transforms

from math import pi

from PIL import Image, ImageDraw

from scipy import io
#from modules.layers import ConvOffset2D

import math
import sys

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
parser.add_argument('--epsilon', type=float, default=10.0)
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
parser.add_argument('--tag', type=str, default='')

args = parser.parse_args()

test_video_keys = PairFromBinDatasetSB(args.test_bins_path, args.parent_dir).bins.keys()

all_video_folders_exist = True
for vid_id in test_video_keys:
    if not os.path.exists('./video_' + args.tag + '_' + vid_id + '/'):
        all_video_folders_exist = False

if not all_video_folders_exist:
    os.system("python train_bins_dnet_" + args.tag + ".py --make_video --weight_path "+args.weight_path)

for vid_id in test_video_keys:

    os.system('ffmpeg -i ./video_' + args.tag + '_' + vid_id + '/image-%0d.png -c:v libx264 -qp 0 -s 640x480 -filter:v fps=90 sample_videos/out_'+args.tag+'_'+vid_id+'.mp4')



#ffmpeg -r 60 -f image2 -s 1920x1080 -i pic%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
#ffmpeg -r 1/5 -i img%00d.png -c:v libx264 -vf "fps=25,format=yuv420p" out.mp4
