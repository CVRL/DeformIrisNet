import numpy as np
import torch
import os
import csv
import math
import random

import cv2
from argparse import ArgumentParser
import torch.nn as nn

from torch.optim import Adam, SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torch.nn.functional import interpolate

from dataset import PairFromBinDataset, AllPairsDataset, PairMinMaxBinDataset
from network import UNet, SIFLayerMask, TVLoss, VGGPerceptualLoss
from scipy import io
import shutil
import datetime
import os
import re
from typing import List, Optional, Tuple, Union
from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg

import click
import dnnlib
import PIL.Image

import legacy
import pickle
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
torch.backends.cudnn.deterministic = True

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.
    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')
    
def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

def img_transform(img_tensor):
    img_t = img_tensor[0]
    img_t = np.clip(img_t.clone().detach().cpu().numpy() * 255, 0, 255)
    img = Image.fromarray(img_t.astype(np.uint8))
    return img

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.NLLLoss(weight)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, outputs, targets):
        return self.loss(self.softmax(outputs), targets)

class matchSIFCode(nn.Module):
    def __init__(self, filter_size, num_filters, max_shift):
        super().__init__()
        self.loss = nn.SmoothL1Loss()
        self.margin = int(math.ceil(filter_size/2))
        self.num_filters = num_filters
        self.max_shift = max_shift
        
    def forward(self, outputs, targets, masks):
        self.code1 = outputs[:, margin:-margin, :, :]
        self.code2 = outputs[:, margin:-margin, :, :]
        self.mask = masks[:, margin:-margin, :]
        scoreC = torch.zeros((self.num_filters, 2*self.max_shift+1))
        for shift in range(-self.max_shift, self.max_shift+1):
            andMasks = torch.logical_and(self.mask, torch.roll(self.mask, shift, dims=2))
            xorCodes = torch.logical_xor(self.code1, torch.roll(self.code2, shift, dims=2))
            xorCodesMasked = torch.logical_and(xorCodes, torch.tile(torch.unsqueeze(andMasks,dim=3),self.num_filters))
            scoreC[:,shift] = torch.sum(xorCodesMasked, dims=(1,2)) / torch.sum(andMasks)
        return torch.min(scoreC)

class RadialCircularCorrelationLoss(nn.Module):
    def __init__(self, polar_height, polar_width, device):
        super().__init__()
        self.loss = nn.SmoothL1Loss()
        self.polar_height = polar_height
        self.polar_width = polar_width
        self.device = device
    
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
        if mask is not None:
            mask_mean = (mask.max() + mask.min())/2
            mask[mask<=mask_mean] = 0
            mask[mask>mask_mean] = 1
            mask_polar = self.grid_sample(torch.tensor(mask).float(), grid_sample_mat)
            one = torch.ones(mask_polar.shape)
            zero = torch.zeros(mask_polar.shape)
            mask_polar = torch.where(mask_polar > 0.5, one, zero)
        else:
            mask_polar = None

        return image_polar, mask_polar
    
    def forward(self, outputs, targets, pupil_xyrs, iris_xyrs):
        outputs_polar, _ = self.cartToPol(outputs, None, pupil_xyrs, iris_xyrs)
        targets_polar, _ = self.cartToPol(targets, None, pupil_xyrs, iris_xyrs)
        shape = outputs_polar.shape
        outputs_polar = outputs_polar.reshape(-1, shape[2], shape[3])
        targets_polar = targets_polar.reshape(-1, shape[2], shape[3])
        outputs_polar
        outputs_gram_1 = torch.bmm(torch.transpose(outputs_polar, 1, 2), outputs_polar)
        targets_gram_1 = torch.bmm(torch.transpose(targets_polar, 1, 2), targets_polar)
        outputs_gram_2 = torch.bmm(outputs_polar, torch.transpose(outputs_polar, 1, 2))
        targets_gram_2 = torch.bmm(targets_polar, torch.transpose(targets_polar, 1, 2))
        return (self.loss(outputs_gram_1, targets_gram_1) + self.loss(outputs_gram_2, targets_gram_2))/2

class CorrelationLoss(nn.Module):
    def __init__(self, polar_height, polar_width, device):
        super().__init__()
        self.loss = nn.SmoothL1Loss()
        self.polar_height = polar_height
        self.polar_width = polar_width
        self.device = device
    
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
    
    def forward(self, outputs, targets, pupil_xyrs, iris_xyrs):
        outputs_polar, _ = self.cartToPol(outputs, masks, pupil_xyrs, iris_xyrs)
        targets_polar, _ = self.cartToPol(targets, None, pupil_xyrs, iris_xyrs)
        shape = outputs_polar.shape
        outputs_polar = outputs_polar.reshape(-1, 1, shape[2]*shape[3])
        targets_polar = targets_polar.reshape(-1, 1, shape[2]*shape[3])
        outputs_gram = torch.bmm(torch.transpose(outputs_polar, 1, 2), outputs_polar)
        targets_gram = torch.bmm(torch.transpose(targets_polar, 1, 2), targets_polar)
        return self.loss(outputs_gram, targets_gram)

class DenseLinearNetwork(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_layer = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                         nn.PReLU(num_parameters=512)
                                         ) 
        self.linlayers_list = []
        self.actlayers_list = []
        for i in range(self.n_layers):
            self.linlayers_list.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.actlayers_list.append(nn.PReLU(num_parameters=512))
        
        self.linlayers_list = nn.ModuleList(self.linlayers_list)
        self.actlayers_list = nn.ModuleList(self.actlayers_list)
        
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x):
        x1 = self.input_layer(x)
        lin_outputs = []
        act_outputs = [x1]
        for i in range(self.n_layers):
            lin_outputs.append(self.linlayers_list[i](sum(act_outputs)))
            act_outputs.append(self.actlayers_list[i](lin_outputs[-1]))
        return self.output_layer(act_outputs[-1])
''' 
def plot_grad_flow(named_parameters, tag):
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(tag+'_gradient_flow.png')
'''
def plot_grad_flow(named_parameters, tag):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append('.')
            ave_grads.append(p.grad.abs().mean().item())
    #print(ave_grads)
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(tag+'_gradient_flow.png')

def resume(args):
    checkpoint_dir = './cp_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + '/'
    softmax = nn.LogSoftmax(dim=1)
    
    if args.train_min_max:
        train_dataset = PairMinMaxBinDataset(args.train_bins_path, args.parent_dir, input_size=(256,256))
    else:
        train_dataset = PairFromBinDataset(args.train_bins_path, args.parent_dir, input_size=(256,256))
    
    train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    
    if args.val_bins_path:
        if args.train_min_max:
            val_dataset = PairMinMaxBinDataset(args.val_bins_path, args.parent_dir, input_size=(256,256))
        else:
            val_dataset = PairFromBinDataset(args.val_bins_path, args.parent_dir, input_size=(256,256))
        val_dataloader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    else:
        val_dataloader = None
    
    print('Actual Batch Size:', args.batch_size)
    print('No. of training batches', len(train_dataloader))
    if val_dataloader is not None:
        print('No. of validation batches', len(val_dataloader))
    
    print('Virtual Batch Size:', args.batch_size * args.virtual_batch_mult)
    
    
    # Declare Models
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    epoch_num = int(args.weight_path.split('/')[-1].split('-')[0])
    
    if not args.train_gen or epoch_num < 5 or args.no_load_gen:
        print('Loading networks from "%s"...' % args.network_pkl)
        with dnnlib.util.open_url(args.network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    else:
        print('Loading trained generator ...')
        G_parent_dir = '/'.join(args.weight_path.split('/')[:-1])
        G_filename = '.'.join(args.weight_path.split('/')[-1].split('.')[:-1])+'-G.pth'
        G = torch.load(os.path.join(G_parent_dir, G_filename), map_location=device).to(device)
        
    conv_map_net = torch.load(args.weight_path, map_location=device).to(device)

    if args.use_mask_loss:
        maskLossModel = UNet(num_classes=2, num_channels=1).to(device)
        maskLossModel.load_state_dict(torch.load(args.mask_model_path, map_location=device))
        maskLossModel.eval()
    
    if args.use_sif_loss:
        filter_mat = io.loadmat(args.sif_filter_path)['ICAtextureFilters']
        sifLossModel = SIFLayerMask(polar_height = 64, polar_width = 512, filter_mat = filter_mat, device=device).to(device)
        sifLossModel.eval()
    
    if args.use_tv_loss:
        tvLossModel = TVLoss(p=2)
    
    if args.use_perceptual_loss:
        vggLossModel = VGGPerceptualLoss(device)
    
    if not args.only_mapping:
        if args.optim_type == 'adam':
            optimizer = Adam(list(conv_map_net.parameters()) + list(G.parameters()), lr = args.lr)
        elif args.optim_type == 'sgd':
            optimizer = SGD(list(conv_map_net.parameters()) + list(G.parameters()), lr=args.lr, momentum=0.9)
        elif args.optim_type == 'cyclic_lr':
            optimizer = SGD(list(conv_map_net.parameters()) + list(G.parameters()), lr=args.lr, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01*args.lr, max_lr=args.lr)
    else:
        if args.optim_type == 'adam':
            optimizer = Adam(list(conv_map_net.parameters()) + list(G.mapping.parameters()), lr = args.lr)
        elif args.optim_type == 'sgd':
            optimizer = SGD(list(conv_map_net.parameters()) + list(G.mapping.parameters()), lr=args.lr, momentum=0.9)
        elif args.optim_type == 'cyclic_lr':
            optimizer = SGD(list(conv_map_net.parameters()) + list(G.mapping.parameters()), lr=args.lr, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01*args.lr, max_lr=args.lr)
    #print(len(optimizer.param_groups))
    
    best_val_loss = float('inf')
    print('Starting Training...')
    sample_no = 0
    for epoch in range(1, args.num_epochs+1):
        
        conv_map_net.train()
        
        G.mapping.train()
        G.synthesis.train()

        epoch_loss = []
        epoch_mse_loss = []
        epoch_mask_loss = []
        epoch_sif_loss = []
        epoch_tv_loss = []
        epoch_percept_loss = []

        if epoch != 1:
            train_dataloader.dataset.reset()
                        
        vb_count = 0
        vb_loss = 0
        vb_batch = -1                     
            
        for batch, data in enumerate(train_dataloader):
            
            small_imgs = data['small_img']
            small_masks = data['small_mask']
            
            big_imgs = data['big_img']
            big_masks = data['big_mask']
            
            small_img_pxyr = data['small_img_pxyr']
            small_img_ixyr = data['small_img_ixyr']
            
            big_img_pxyr = data['big_img_pxyr'].requires_grad_(False)
            big_img_ixyr = data['big_img_ixyr'].requires_grad_(False)
            
            inp = Variable(torch.cat([small_imgs, big_masks], dim=1)).to(device, non_blocking=True)
            inp_mask = Variable(data['big_mask']).to(device, non_blocking=True)
            
            tar = Variable(big_imgs).to(device, non_blocking=True)

            z = conv_map_net(inp)
            if hasattr(G.synthesis, 'input'):
                m = make_transform(args.translate, args.rotate)
                m = np.linalg.inv(m)
                G.synthesis.input.transform.copy_(torch.from_numpy(m))
            label = torch.zeros([1, G.c_dim], device=device)
            out = G(z, label, truncation_psi=1, noise_mode='const')
            
            out_norm = (out+1)/2
            tar_norm = (tar+1)/2
            inp_mask_norm = ((inp_mask+1)/2)                    
            
            mse_loss = nn.L1Loss()(out, tar.requires_grad_(False))
            epoch_mse_loss.append(args.alpha * mse_loss.item())
            
            if args.use_tv_loss:
                tv_loss = tvLossModel(out_norm)
                epoch_tv_loss.append(args.epsilon * tv_loss)
            
            if args.use_perceptual_loss:
                percept_loss = vggLossModel(out, tar.requires_grad_(False))
                epoch_percept_loss.append(args.sigma * percept_loss) 
            
            mask_out_logprob = maskLossModel(interpolate(out_norm, size=(240, 320), mode='bilinear'))
            mask_inp_logprob = maskLossModel(interpolate(tar_norm, size=(240, 320), mode='bilinear')).detach().requires_grad_(False)
              
            if args.use_mask_loss:
                if args.mask_bce:
                    mask_loss = nn.BCELoss()(nn.Softmax(dim=1)(mask_out_logprob), nn.Softmax(dim=1)(mask_inp_logprob))
                else:
                    mask_loss = nn.L1Loss()(mask_out_logprob, mask_inp_logprob)
                    #inp_mask_norm_resized = interpolate(inp_mask_norm, size=(240, 320), mode='nearest').requires_grad_(False)
                    #mask_shape = inp_mask_norm_resized.shape
                    #mask_loss = CrossEntropyLoss2d()(mask_out_logprob, inp_mask_norm_resized.reshape(mask_shape[0], mask_shape[2], mask_shape[3]).long())
                epoch_mask_loss.append(args.beta * mask_loss.item())
            else:
                mask_loss = 0.
            
            if args.use_sif_loss:
                mask_out = torch.argmax(mask_out_logprob, dim=1).reshape(-1,1,240,320)
                sif_out, sif_out_gram, out_img_polar, out_mask_polar = sifLossModel(interpolate(out_norm, size=(240, 320), mode='bilinear'), big_img_pxyr, big_img_ixyr, mask=interpolate(inp_mask_norm, size=(240, 320), mode='nearest') )
                sif_tar, sif_tar_gram, tar_img_polar, tar_mask_polar = sifLossModel(interpolate(tar_norm, size=(240, 320), mode='bilinear'), big_img_pxyr, big_img_ixyr, mask=interpolate(inp_mask_norm, size=(240, 320), mode='nearest') )
                #print(sif_tar.min(), sif_tar.max())
                if batch == 0 and epoch == 1:
                    for bi in range(out_img_polar.shape[0]):
                        cv2.imwrite('samples_polar/out_img_sample'+str(bi)+'.png', out_img_polar[bi][0].clone().detach().cpu().numpy())
                        cv2.imwrite('samples_polar/tar_img_sample'+str(bi)+'.png', tar_img_polar[bi][0].clone().detach().cpu().numpy())
                sif_tar_ng = sif_tar.clone().detach().requires_grad_(False)
                one = torch.ones(sif_tar.shape).to(device)
                zero = torch.zeros(sif_tar_ng.shape).to(device)
                sif_tar_binary = torch.where(sif_tar_ng > 0, one, zero).requires_grad_(False)
                tar_mask_polar_rep = torch.cat([tar_mask_polar]*7, dim=1).requires_grad_(False)
                sif_shape = sif_out.shape
                sif_out_masked = (nn.Sigmoid()(sif_out) * tar_mask_polar_rep).requires_grad_(True)
                sif_tar_masked = (nn.Sigmoid()(sif_tar_ng) * tar_mask_polar_rep).requires_grad_(False)
                sif_tar_binary_masked = torch.clamp((sif_tar_binary * tar_mask_polar_rep).requires_grad_(False), 0, 1).float()
                if args.sif_bce:
                    #sif_loss = nn.BCELoss()((nn.Flatten()(sif_out_masked)), nn.Flatten()(sif_tar_binary_masked))
                    sif_loss = nn.BCELoss()(sif_out_masked, sif_tar_masked)
                else:
                    #sif_loss = nn.SmoothL1Loss()(sif_out_masked, sif_tar_masked)
                    #sif_loss = nn.MSELoss()(sif_out_masked, sif_tar_masked)
                    sif_loss = nn.L1Loss()(sif_out * tar_mask_polar_rep, sif_tar_ng * tar_mask_polar_rep)
                if args.sif_gram:
                    sif_loss += nn.SmoothL1Loss()(sif_out_gram, sif_tar_gram)
                epoch_sif_loss.append(args.gamma * sif_loss.item())
            else:
                sif_loss = 0.
            
            loss = args.alpha * mse_loss + args.beta * mask_loss + args.gamma * sif_loss + args.epsilon * tv_loss + args.sigma * percept_loss
            vb_loss += loss.item()
            loss.backward()
            vb_count += 1
            
            if vb_count == args.virtual_batch_mult:
                optimizer.step()
                if args.optim_type == 'cyclic_lr':
                    scheduler.step()
                epoch_loss.append(vb_loss/args.virtual_batch_mult)
                vb_loss = 0
                vb_count = 0
                vb_batch += 1
                if vb_batch % args.log_batch == 0:
                    plot_grad_flow(conv_map_net.named_parameters(), 'conv_map_net')
                    #plot_grad_flow(G.named_parameters(), 'G')
                optimizer.zero_grad()
            
            if vb_batch % args.log_batch == 0 and vb_count == 0:
                train_loss_average = sum(epoch_loss) / len(epoch_loss)
                mse_loss_average = sum(epoch_mse_loss) / len(epoch_mse_loss)
                loss_string = "Train loss: {aver} (epoch: {epoch}, batch: {batch}) Direct loss: {mse}".format(aver = train_loss_average, epoch = epoch, batch = vb_batch, mse = mse_loss_average)
                
                '''
                if args.use_corr_loss or args.use_radial_corr_loss:
                    corr_loss_average = sum(epoch_corr_loss) / len(epoch_corr_loss)
                    loss_string += ", Gram loss: {corr}".format(corr = corr_loss_average)
                '''             
                
                if args.use_tv_loss:
                    tv_loss_average = sum(epoch_tv_loss) / len(epoch_tv_loss)
                    loss_string += ", TV loss: {tv}".format(tv = tv_loss_average)
                
                if args.use_perceptual_loss:
                    percept_loss_average = sum(epoch_percept_loss) / len(epoch_percept_loss)
                    loss_string += ", Perceptual loss: {percept}".format(percept = percept_loss_average)
                
                if args.use_mask_loss:
                    mask_loss_average = sum(epoch_mask_loss) / len(epoch_mask_loss)
                    loss_string += ", Mask loss: {mask}".format(mask = mask_loss_average)
                
                if args.use_sif_loss:
                    sif_loss_average = sum(epoch_sif_loss) / len(epoch_sif_loss)
                    loss_string += ", SIF loss: {sif}".format(sif = sif_loss_average)
                    sif_out_binary = torch.where(nn.Sigmoid()(sif_out).clone().detach().requires_grad_(False) > 0.5, one, zero)
                    sif_out_binary_masked = (sif_out_binary * tar_mask_polar_rep).requires_grad_(False)
                    loss_string += ", SIF bit diff: {sif_bit}%".format(sif_bit = torch.mean(torch.abs(sif_out_binary_masked - sif_tar_binary_masked).flatten()) * 100)
                '''
                if sample_no < 10:
                    small_img_t = interpolate((small_imgs + 1)/2, size=(240, 320), mode='bilinear')
                    big_img_t = interpolate((big_imgs + 1)/2, size=(240, 320), mode='bilinear')
                    small_mask_t = interpolate((small_masks + 1)/2, size=(240, 320), mode='nearest')
                    big_mask_t = interpolate((big_masks + 1)/2, size=(240, 320), mode='nearest')
                    
                    for b in range(small_img_t.shape[0]):
                    
                        s_img = img_transform(small_img_t[b])
                        b_img = img_transform(big_img_t[b])
                        s_mask = img_as_bool((small_mask_t[b][0].clone().detach().cpu().numpy() * 255).astype(np.uint8))
                        b_mask = img_as_bool((big_mask_t[b][0].clone().detach().cpu().numpy() * 255).astype(np.uint8))
                        
                        s_pxyr = np.around(small_img_pxyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                        s_ixyr = np.around(small_img_ixyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                        b_pxyr = np.around(big_img_pxyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                        b_ixyr = np.around(big_img_ixyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                        
                        s_img_c = s_img.convert('RGB')
                        b_img_c = b_img.convert('RGB')
                        
                        s_draw = ImageDraw.Draw(s_img_c)
                        s_draw.ellipse((s_pxyr[0]-s_pxyr[2], s_pxyr[1]-s_pxyr[2], s_pxyr[0]+s_pxyr[2], s_pxyr[1]+s_pxyr[2]), outline ="red")
                        s_draw.ellipse((s_ixyr[0]-s_ixyr[2], s_ixyr[1]-s_ixyr[2], s_ixyr[0]+s_ixyr[2], s_ixyr[1]+s_ixyr[2]), outline ="blue")
                        b_draw = ImageDraw.Draw(b_img_c)
                        b_draw.ellipse((b_pxyr[0]-b_pxyr[2], b_pxyr[1]-b_pxyr[2], b_pxyr[0]+b_pxyr[2], b_pxyr[1]+b_pxyr[2]), outline ="red")
                        b_draw.ellipse((b_ixyr[0]-b_ixyr[2], b_ixyr[1]-b_ixyr[2], b_ixyr[0]+b_ixyr[2], b_ixyr[1]+b_ixyr[2]), outline ="blue")
                        
                        all_imgs = get_concat_v(get_concat_v(s_img_c, Image.fromarray(s_mask.astype(np.uint8) * 255)), get_concat_v(b_img_c, Image.fromarray(b_mask.astype(np.uint8) * 255)))
                        all_imgs.save('samples/img_sample_'+str(batch)+'_'+str(b)+'.png')
                        
                    sample_no += 1
                '''   
                
                print(loss_string)
                if args.log_file is not None:
                    with open(args.log_file, 'a') as f:
                        f.write(loss_string + '\n')
                
                    
        
        if val_dataloader is not None:
            conv_map_net.eval()
            G.mapping.eval()
            G.synthesis.eval()
            G.eval()
            val_loss_average = 0
            val_bit_diff_average = 0
            with torch.no_grad():
                for i in range(args.val_repeats):
                    val_dataloader.dataset.reset()
                    val_epoch_loss = []
                    val_bit_diff = []
                    for batch, data in enumerate(val_dataloader):
                    
                        small_imgs = data['small_img']
                        small_masks = data['small_mask']
                        
                        big_imgs = data['big_img']
                        big_masks = data['big_mask']
                        
                        small_img_pxyr = data['small_img_pxyr']
                        small_img_ixyr = data['small_img_ixyr']
                        
                        big_img_pxyr = data['big_img_pxyr'].requires_grad_(False)
                        big_img_ixyr = data['big_img_ixyr'].requires_grad_(False)
                        
                        inp = Variable(torch.cat([small_imgs, big_masks], dim=1)).to(device, non_blocking=True)
                        inp_mask = Variable(data['big_mask']).to(device, non_blocking=True)
                        
                        tar = Variable(big_imgs).to(device, non_blocking=True)
            
                        z = conv_map_net(inp)
                        if hasattr(G.synthesis, 'input'):
                            m = make_transform(args.translate, args.rotate)
                            m = np.linalg.inv(m)
                            G.synthesis.input.transform.copy_(torch.from_numpy(m))
                        label = torch.zeros([1, G.c_dim], device=device)
                        out = G(z, label, truncation_psi=1, noise_mode='const')
                        
                        out_norm = (out+1)/2
                        tar_norm = (tar+1)/2
                        inp_mask_norm = ((inp_mask+1)/2)                    
                        
                        mse_loss = nn.L1Loss()(out, tar.requires_grad_(False))
                        epoch_mse_loss.append(args.alpha * mse_loss.item())
                        
                        if args.use_tv_loss:
                            tv_loss = tvLossModel(out_norm)
                        
                        if args.use_perceptual_loss:
                            percept_loss = vggLossModel(out, tar.requires_grad_(False))
                                    
                        mask_out_logprob = maskLossModel(interpolate(out_norm, size=(240, 320), mode='bilinear'))
                        mask_inp_logprob = maskLossModel(interpolate(tar_norm, size=(240, 320), mode='bilinear')).detach().requires_grad_(False)
              
                        if args.use_mask_loss:
                            if args.mask_bce:
                                mask_loss = nn.BCELoss()(nn.Softmax(dim=1)(mask_out_logprob), nn.Softmax(dim=1)(mask_inp_logprob))
                            else:
                                mask_loss = nn.L1Loss()(mask_out_logprob, mask_inp_logprob)
                                  
                                #mask_loss = nn.MSELoss()(mask_out_logprob, mask_inp_logprob)
                                
                                #inp_mask_norm_resized = interpolate(inp_mask_norm, size=(240, 320), mode='nearest').requires_grad_(False)
                                #mask_shape = inp_mask_norm_resized.shape
                                #mask_loss = CrossEntropyLoss2d()(mask_out_logprob, inp_mask_norm_resized.reshape(mask_shape[0], mask_shape[2], mask_shape[3]).long())
                        else:
                            mask_loss = 0.
                        
                        if args.use_sif_loss:
                            mask_out = torch.argmax(mask_out_logprob, dim=1).reshape(-1,1,240,320)
                            sif_out, sif_out_gram, out_img_polar, out_mask_polar = sifLossModel(interpolate(out_norm, size=(240, 320), mode='bilinear'), big_img_pxyr, big_img_ixyr, mask=interpolate(inp_mask_norm, size=(240, 320), mode='nearest') )
                            sif_tar, sif_tar_gram, tar_img_polar, tar_mask_polar = sifLossModel(interpolate(tar_norm, size=(240, 320), mode='bilinear'), big_img_pxyr, big_img_ixyr, mask=interpolate(inp_mask_norm, size=(240, 320), mode='nearest') )
                            if batch == 0 and epoch == 1:
                                for bi in range(out_img_polar.shape[0]):
                                    cv2.imwrite('samples_polar/out_img_sample'+str(bi)+'.png', out_img_polar[bi][0].clone().detach().cpu().numpy())
                                    cv2.imwrite('samples_polar/tar_img_sample'+str(bi)+'.png', tar_img_polar[bi][0].clone().detach().cpu().numpy())
                            sif_tar_ng = sif_tar.clone().detach().requires_grad_(False)
                            one = torch.ones(sif_tar.shape).to(device)
                            zero = torch.zeros(sif_tar_ng.shape).to(device)
                            sif_tar_binary = torch.where(sif_tar_ng > 0, one, zero).requires_grad_(False)
                            tar_mask_polar_rep = torch.cat([tar_mask_polar]*7, dim=1).requires_grad_(False)
                            sif_shape = sif_out.shape
                            sif_out_masked = (nn.Sigmoid()(sif_out) * tar_mask_polar_rep).requires_grad_(True)
                            sif_tar_binary_masked = torch.clamp((sif_tar_binary * tar_mask_polar_rep).requires_grad_(False), 0, 1).float()
                            sif_tar_masked = (nn.Sigmoid()(sif_tar_ng) * tar_mask_polar_rep).requires_grad_(False)
                            if args.sif_bce:
                                #sif_loss = nn.BCELoss()((nn.Flatten()(sif_out_masked)), nn.Flatten()(sif_tar_binary_masked))
                                sif_loss = nn.BCELoss()(sif_out_masked, sif_tar_masked)
                            else:
                                #sif_loss = nn.SmoothL1Loss()(sif_out_masked, sif_tar_masked)
                                sif_loss = nn.L1Loss()(sif_out * tar_mask_polar_rep, sif_tar_ng * tar_mask_polar_rep)
                            if args.sif_gram:
                                sif_loss += nn.SmoothL1Loss()(sif_out_gram, sif_tar_gram)
                            sif_out_binary = torch.where(nn.Sigmoid()(sif_out).clone().detach().requires_grad_(False) > 0.5, one, zero)
                            sif_out_binary_masked = (sif_out_binary * tar_mask_polar_rep).requires_grad_(False)
                            val_bit_diff.append(torch.mean(torch.abs(sif_out_binary_masked - sif_tar_binary_masked).flatten()) * 100)
                        else:
                            sif_loss = 0.

                        loss = args.alpha * mse_loss + args.beta * mask_loss + args.gamma * sif_loss + args.epsilon * tv_loss + args.sigma * percept_loss
                        
                        val_epoch_loss.append(loss.item())
                        
                    val_loss_average += (sum(val_epoch_loss) / len(val_epoch_loss))
                    val_bit_diff_average += (sum(val_bit_diff) / len(val_bit_diff))
                
                val_loss_average /= args.val_repeats
                val_bit_diff_average /= args.val_repeats
                print("Val loss: {aver}, Val bit diff: {bit_diff} (epoch: {epoch})".format(aver = val_loss_average, bit_diff = val_bit_diff_average, epoch = epoch))
                if args.log_file is not None:
                    with open(args.log_file, 'a') as f:
                        f.write("Val loss: {aver}, Val bit diff: {bit_diff} (epoch: {epoch})".format(aver = val_loss_average, bit_diff = val_bit_diff_average, epoch = epoch) + '\n')
                
                if val_loss_average < best_val_loss:
                    best_val_loss = val_loss_average
                    if not os.path.exists(checkpoint_dir):
                        os.mkdir(checkpoint_dir)
                    filename = checkpoint_dir + "{epoch:04}-val_loss-{val_loss}-val_bit_diff-{bit_diff}.pth".format(epoch = epoch, use_mask_loss=args.use_mask_loss, use_sif_loss=args.use_sif_loss, val_loss=best_val_loss, bit_diff=val_bit_diff_average)
                    torch.save(conv_map_net, filename)
                    G_filename = checkpoint_dir + "{epoch:04}-val_loss-{val_loss}-val_bit_diff-{bit_diff}-G.pth".format(epoch = epoch, use_mask_loss=args.use_mask_loss, use_sif_loss=args.use_sif_loss, val_loss=best_val_loss, bit_diff=val_bit_diff_average)
                    torch.save(G, G_filename)
    return True

def train(args):

    checkpoint_dir = './cp_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + '/'
    softmax = nn.LogSoftmax(dim=1)
    # Declare Dataloaders
    if args.train_min_max:
        train_dataset = PairMinMaxBinDataset(args.train_bins_path, args.parent_dir, input_size=(256,256))
    else:
        train_dataset = PairFromBinDataset(args.train_bins_path, args.parent_dir, input_size=(256,256))
    
    train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    
    if args.val_bins_path:
        if args.train_min_max:
            val_dataset = PairMinMaxBinDataset(args.val_bins_path, args.parent_dir, input_size=(256,256))
        else:
            val_dataset = PairFromBinDataset(args.val_bins_path, args.parent_dir, input_size=(256,256))
        val_dataloader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    else:
        val_dataloader = None
    
    print('No. of training batches', len(train_dataloader))
    if val_dataloader is not None:
        print('No. of validation batches', len(val_dataloader))
    
    # Declare Models
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    
    print('Loading networks from "%s"...' % args.network_pkl)
    with dnnlib.util.open_url(args.network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        
    #conv_map_net = models.resnet50(pretrained=False)
    #conv_map_net.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #conv_map_net.fc = nn.Linear(2048, G.z_dim)
    #conv_map_net = conv_map_net.to(device)
    
    #conv_map_net = models.resnet18(pretrained=False)
    #conv_map_net.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #conv_map_net.fc = DenseLinearNetwork(n_layers=7, input_dim=512, hidden_dim=512, output_dim=G.z_dim)
    
    print('Using DenseNet161...')
    conv_map_net = models.densenet161(pretrained=False)
    conv_map_net.features.conv0 = nn.Conv2d(2, 96, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    conv_map_net.classifier = nn.Linear(in_features=2208, out_features=G.z_dim, bias=True)
        
    print(conv_map_net)
    conv_map_net = conv_map_net.to(device)

    if args.use_tv_loss:
        tvLossModel = TVLoss(p=2)
    
    if args.use_perceptual_loss:
        vggLossModel = VGGPerceptualLoss(device)

    if args.use_mask_loss:
        maskLossModel = UNet(num_classes=2, num_channels=1).to(device)
        maskLossModel.load_state_dict(torch.load(args.mask_model_path, map_location=device))
        maskLossModel.eval()
    
    if args.use_sif_loss:
        filter_mat = io.loadmat(args.sif_filter_path)['ICAtextureFilters']
        sifLossModel = SIFLayerMask(polar_height = 64, polar_width = 512, filter_mat = filter_mat, device=device).to(device)
        sifLossModel.eval()
    
    if args.optim_type == 'adam':
        optimizer = Adam(conv_map_net.parameters(recurse=True), lr = args.lr)
    elif args.optim_type == 'sgd':
        optimizer = SGD(conv_map_net.parameters(recurse=True), lr = args.lr, momentum=0.9)
    optimizer.zero_grad()
    #print(len(optimizer.param_groups))
    
    best_val_loss = float('inf')
    print('Starting Training...')
    sample_no = 0
    for epoch in range(1, args.num_epochs+1):
    
        conv_map_net.train()
        
        epoch_loss = []
        epoch_mse_loss = []
        epoch_mask_loss = []
        epoch_sif_loss = []
        epoch_tv_loss = []
        epoch_percept_loss = []

        if epoch != 1:
            train_dataloader.dataset.reset()
        
        if epoch == 10:
            print('Adding G mapping ...')
            if args.log_file is not None:
                with open(args.log_file, 'a') as f:
                    f.write('Adding G mapping...'+ '\n')
            optimizer.add_param_group({'params': G.mapping.parameters(recurse=True)})
        elif epoch == 20 and args.train_gen:
            print('Adding G synthesis ...')
            if args.log_file is not None:
                with open(args.log_file, 'a') as f:
                    f.write('Adding G mapping...'+ '\n')                
            optimizer.add_param_group({'params': G.synthesis.parameters(recurse=True)})
                 
        if epoch >= 10:
            G.mapping.train()
        elif epoch >= 20:
            G.mapping.train()
            G.synthesis.train()
            G.train()
        
        vb_count = 0
        vb_loss = 0
        vb_batch = -1 
        for batch, data in enumerate(train_dataloader):
        
            small_imgs = data['small_img']
            small_masks = data['small_mask']
            
            big_imgs = data['big_img']
            big_masks = data['big_mask']
            
            small_img_pxyr = data['small_img_pxyr']
            small_img_ixyr = data['small_img_ixyr']
            
            big_img_pxyr = data['big_img_pxyr']
            big_img_ixyr = data['big_img_ixyr']
            
            inp = Variable(torch.cat([small_imgs, big_masks], dim=1)).to(device)
            inp_mask = Variable(data['big_mask']).to(device)
            
            tar = Variable(big_imgs).to(device)

            z = conv_map_net(inp)
            if hasattr(G.synthesis, 'input'):
                m = make_transform(args.translate, args.rotate)
                m = np.linalg.inv(m)
                G.synthesis.input.transform.copy_(torch.from_numpy(m))
            label = torch.zeros([1, G.c_dim], device=device)
            out = G(z, label, truncation_psi=1, noise_mode='const')
            
            epoch_mse_loss.append(args.alpha * mse_loss.item())
            
            out_norm = (out+1)/2
            tar_norm = (tar+1)/2
            inp_mask_norm = ((inp_mask+1)/2)                    
            
            mse_loss = nn.L1Loss()(out, tar.requires_grad_(False))
            epoch_mse_loss.append(args.alpha * mse_loss.item())
            
            if args.use_tv_loss:
                tv_loss = tvLossModel(out_norm)
                epoch_tv_loss.append(args.epsilon * tv_loss)
            
            if args.use_perceptual_loss:
                percept_loss = vggLossModel(out, tar.requires_grad_(False))
                epoch_percept_loss.append(args.sigma * percept_loss) 
            
            mask_out_logprob = maskLossModel(interpolate(out_norm, size=(240, 320), mode='bilinear'))
            mask_inp_logprob = maskLossModel(interpolate(tar_norm, size=(240, 320), mode='bilinear')).detach().requires_grad_(False)
              
            if args.use_mask_loss:
                if args.mask_bce:
                    mask_loss = nn.BCELoss()(nn.Softmax(dim=1)(mask_out_logprob), nn.Softmax(dim=1)(mask_inp_logprob))
                else:
                    mask_loss = nn.L1Loss()(mask_out_logprob, mask_inp_logprob)
                    
                    #mask_loss = nn.MSELoss()(mask_out_logprob, mask_inp_logprob)
                    
                    #inp_mask_norm_resized = interpolate(inp_mask_norm, size=(240, 320), mode='nearest').requires_grad_(False)
                    #mask_shape = inp_mask_norm_resized.shape
                    #mask_loss = CrossEntropyLoss2d()(mask_out_logprob, inp_mask_norm_resized.reshape(mask_shape[0], mask_shape[2], mask_shape[3]).long())
                epoch_mask_loss.append(args.beta * mask_loss.item())
            else:
                mask_loss = 0.
            
            if args.use_sif_loss:
                sif_out, sif_out_gram, out_img_polar, out_mask_polar = sifLossModel(interpolate(out_norm, size=(240, 320), mode='bilinear'), big_img_pxyr, big_img_ixyr, mask=interpolate(inp_mask_norm, size=(240, 320), mode='nearest') )
                sif_tar, sif_tar_gram, tar_img_polar, tar_mask_polar = sifLossModel(interpolate(tar_norm, size=(240, 320), mode='bilinear'), big_img_pxyr, big_img_ixyr, mask=interpolate(inp_mask_norm, size=(240, 320), mode='nearest') )
                if batch == 0 and epoch == 1:
                    for bi in range(out_img_polar.shape[0]):
                        cv2.imwrite('samples_polar/out_img_sample'+str(bi)+'.png', out_img_polar[bi][0].clone().detach().cpu().numpy())
                        cv2.imwrite('samples_polar/tar_img_sample'+str(bi)+'.png', tar_img_polar[bi][0].clone().detach().cpu().numpy())
                sif_tar_ng = sif_tar.clone().detach().requires_grad_(False)
                one = torch.ones(sif_tar.shape).to(device)
                zero = torch.zeros(sif_tar_ng.shape).to(device)
                sif_tar_binary = torch.where(sif_tar_ng > 0, one, zero).requires_grad_(False)
                tar_mask_polar_rep = torch.cat([tar_mask_polar]*7, dim=1).requires_grad_(False)
                sif_shape = sif_out.shape
                sif_out_masked = (nn.Sigmoid()(sif_out) * tar_mask_polar_rep).requires_grad_(True)
                sif_tar_binary_masked = torch.clamp((sif_tar_binary * tar_mask_polar_rep).requires_grad_(False), 0, 1).float()
                if args.sif_bce:
                    #print(sif_out_masked.shape, sif_tar_binary_masked.shape)
                    #sif_loss = nn.BCELoss()((nn.Flatten()(sif_out_masked)), nn.Flatten()(sif_tar_binary_masked))
                    sif_tar_masked = (nn.Sigmoid()(sif_tar_ng) * tar_mask_polar_rep).requires_grad_(False)
                    sif_loss = nn.BCELoss()(sif_out_masked, sif_tar_masked)
                else:
                    #sif_loss = nn.SmoothL1Loss()(nn.Tanh()(sif_out), nn.Tanh()(sif_tar_ng))
                    sif_loss = nn.L1Loss()(sif_out * tar_mask_polar_rep, sif_tar_ng * tar_mask_polar_rep)
                if args.sif_gram:
                    sif_loss += nn.SmoothL1Loss()(sif_out_gram, sif_tar_gram)
                epoch_sif_loss.append(args.gamma * sif_loss.item())
            else:
                sif_loss = 0.
            
            loss = args.alpha * mse_loss + args.beta * mask_loss + args.gamma * sif_loss + args.epsilon * tv_loss + args.sigma * percept_loss
            vb_loss += loss.item()
            loss.backward()
            vb_count += 1
            
            if vb_count == args.virtual_batch_mult:
                optimizer.step()
                if args.optim_type == 'cyclic_lr':
                    scheduler.step()
                epoch_loss.append(vb_loss/args.virtual_batch_mult)
                vb_loss = 0
                vb_count = 0
                vb_batch += 1
                if vb_batch % args.log_batch == 0:
                    plot_grad_flow(conv_map_net.named_parameters(), 'conv_map_net')
                    #plot_grad_flow(G.named_parameters(), 'G')
                optimizer.zero_grad()
            
            if vb_batch % args.log_batch == 0 and vb_count == 0:
                train_loss_average = sum(epoch_loss) / len(epoch_loss)
                mse_loss_average = sum(epoch_mse_loss) / len(epoch_mse_loss)
                loss_string = "Train loss: {aver} (epoch: {epoch}, batch: {batch}) Direct loss: {mse}".format(aver = train_loss_average, epoch = epoch, batch = vb_batch, mse = mse_loss_average)
                
                if args.use_tv_loss:
                    tv_loss_average = sum(epoch_tv_loss) / len(epoch_tv_loss)
                    loss_string += ", TV loss: {tv}".format(tv = tv_loss_average)
                
                if args.use_perceptual_loss:
                    percept_loss_average = sum(epoch_percept_loss) / len(epoch_percept_loss)
                    loss_string += ", Perceptual loss: {percept}".format(percept = percept_loss_average)             
                
                if args.use_mask_loss:
                    mask_loss_average = sum(epoch_mask_loss) / len(epoch_mask_loss)
                    loss_string += ", Mask loss: {mask}".format(mask = mask_loss_average)
                
                if args.use_sif_loss:
                    sif_loss_average = sum(epoch_sif_loss) / len(epoch_sif_loss)
                    loss_string += ", SIF loss: {sif}".format(sif = sif_loss_average)
                    sif_out_binary = torch.where(sif_out_masked.clone().detach().requires_grad_(False) > 0.5, one, zero)
                    loss_string += ", SIF bit diff: {sif_bit}%".format(sif_bit = torch.mean(torch.abs(sif_out_binary * tar_mask_polar_rep - sif_tar_binary_masked).flatten()) * 100)
                '''
                if sample_no < 10:
                    small_img_t = interpolate((small_imgs + 1)/2, size=(240, 320), mode='bilinear')
                    big_img_t = interpolate((big_imgs + 1)/2, size=(240, 320), mode='bilinear')
                    small_mask_t = interpolate((small_masks + 1)/2, size=(240, 320), mode='nearest')
                    big_mask_t = interpolate((big_masks + 1)/2, size=(240, 320), mode='nearest')
                    
                    for b in range(small_img_t.shape[0]):
                    
                        s_img = img_transform(small_img_t[b])
                        b_img = img_transform(big_img_t[b])
                        s_mask = img_as_bool((small_mask_t[b][0].clone().detach().cpu().numpy() * 255).astype(np.uint8))
                        b_mask = img_as_bool((big_mask_t[b][0].clone().detach().cpu().numpy() * 255).astype(np.uint8))
                        
                        s_pxyr = np.around(small_img_pxyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                        s_ixyr = np.around(small_img_ixyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                        b_pxyr = np.around(big_img_pxyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                        b_ixyr = np.around(big_img_ixyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                        
                        s_img_c = s_img.convert('RGB')
                        b_img_c = b_img.convert('RGB')
                        
                        s_draw = ImageDraw.Draw(s_img_c)
                        s_draw.ellipse((s_pxyr[0]-s_pxyr[2], s_pxyr[1]-s_pxyr[2], s_pxyr[0]+s_pxyr[2], s_pxyr[1]+s_pxyr[2]), outline ="red")
                        s_draw.ellipse((s_ixyr[0]-s_ixyr[2], s_ixyr[1]-s_ixyr[2], s_ixyr[0]+s_ixyr[2], s_ixyr[1]+s_ixyr[2]), outline ="blue")
                        b_draw = ImageDraw.Draw(b_img_c)
                        b_draw.ellipse((b_pxyr[0]-b_pxyr[2], b_pxyr[1]-b_pxyr[2], b_pxyr[0]+b_pxyr[2], b_pxyr[1]+b_pxyr[2]), outline ="red")
                        b_draw.ellipse((b_ixyr[0]-b_ixyr[2], b_ixyr[1]-b_ixyr[2], b_ixyr[0]+b_ixyr[2], b_ixyr[1]+b_ixyr[2]), outline ="blue")
                        
                        all_imgs = get_concat_v(get_concat_v(s_img_c, Image.fromarray(s_mask.astype(np.uint8) * 255)), get_concat_v(b_img_c, Image.fromarray(b_mask.astype(np.uint8) * 255)))
                        all_imgs.save('samples/img_sample_'+str(batch)+'_'+str(b)+'.png')
                        
                    sample_no += 1
                '''   
                
                print(loss_string)
                if args.log_file is not None:
                    with open(args.log_file, 'a') as f:
                        f.write(loss_string + '\n')
                
                    
        
        if val_dataloader is not None:
            conv_map_net.eval()
            G.mapping.eval()
            G.synthesis.eval()
            G.eval()
            val_loss_average = 0
            val_bit_diff_average = 0
            with torch.no_grad():
                for i in range(args.val_repeats):
                    val_dataloader.dataset.reset()
                    val_epoch_loss = []
                    val_bit_diff = []
                    for batch, data in enumerate(val_dataloader):
                    
                        small_imgs = data['small_img']
                        small_masks = data['small_mask']
                        
                        big_imgs = data['big_img']
                        big_masks = data['big_mask']
                        
                        small_img_pxyr = data['small_img_pxyr']
                        small_img_ixyr = data['small_img_ixyr']
                        
                        big_img_pxyr = data['big_img_pxyr'].requires_grad_(False)
                        big_img_ixyr = data['big_img_ixyr'].requires_grad_(False)
                        
                        inp = Variable(torch.cat([small_imgs, big_masks], dim=1)).to(device, non_blocking=True)
                        inp_mask = Variable(data['big_mask']).to(device, non_blocking=True)
                        
                        tar = Variable(big_imgs).to(device, non_blocking=True)
            
                        z = conv_map_net(inp)
                        if hasattr(G.synthesis, 'input'):
                            m = make_transform(args.translate, args.rotate)
                            m = np.linalg.inv(m)
                            G.synthesis.input.transform.copy_(torch.from_numpy(m))
                        label = torch.zeros([1, G.c_dim], device=device)
                        out = G(z, label, truncation_psi=1, noise_mode='const')
                        
                        out_norm = (out+1)/2
                        tar_norm = (tar+1)/2
                        inp_mask_norm = ((inp_mask+1)/2)                    
                        
                        mse_loss = nn.L1Loss()(out, tar.requires_grad_(False))
                        epoch_mse_loss.append(args.alpha * mse_loss.item())
                        
                        if args.use_tv_loss:
                            tv_loss = tvLossModel(out_norm)
                        
                        if args.use_perceptual_loss:
                            percept_loss = vggLossModel(out, tar.requires_grad_(False))
                                    
                        mask_out_logprob = maskLossModel(interpolate(out_norm, size=(240, 320), mode='bilinear'))
                        mask_inp_logprob = maskLossModel(interpolate(tar_norm, size=(240, 320), mode='bilinear')).detach().requires_grad_(False)
                        
                        if args.use_mask_loss:
                            if args.mask_bce:
                                mask_loss = nn.BCELoss()(nn.Softmax(dim=1)(mask_out_logprob), nn.Softmax(dim=1)(mask_inp_logprob))
                            else:
                                mask_loss = nn.L1Loss()(mask_out_logprob, mask_inp_logprob)
                                
                                #mask_loss = nn.MSELoss()(mask_out_logprob, mask_inp_logprob)
                                
                                #inp_mask_norm_resized = interpolate(inp_mask_norm, size=(240, 320), mode='nearest').requires_grad_(False)
                                #mask_shape = inp_mask_norm_resized.shape
                                #mask_loss = CrossEntropyLoss2d()(mask_out_logprob, inp_mask_norm_resized.reshape(mask_shape[0], mask_shape[2], mask_shape[3]).long())
                            epoch_mask_loss.append(args.beta * mask_loss.item())
                        else:
                            mask_loss = 0.
                        
                        if args.use_sif_loss:
                            mask_out = torch.argmax(mask_out_logprob, dim=1).reshape(-1,1,240,320)
                            sif_out, sif_out_gram, out_img_polar, out_mask_polar = sifLossModel(interpolate(out_norm, size=(240, 320), mode='bilinear'), big_img_pxyr, big_img_ixyr, mask=interpolate(inp_mask_norm, size=(240, 320), mode='nearest') )
                            sif_tar, sif_tar_gram, tar_img_polar, tar_mask_polar = sifLossModel(interpolate(tar_norm, size=(240, 320), mode='bilinear'), big_img_pxyr, big_img_ixyr, mask=interpolate(inp_mask_norm, size=(240, 320), mode='nearest') )
                            if batch == 0 and epoch == 1:
                                for bi in range(out_img_polar.shape[0]):
                                    cv2.imwrite('samples_polar/out_img_sample'+str(bi)+'.png', out_img_polar[bi][0].clone().detach().cpu().numpy())
                                    cv2.imwrite('samples_polar/tar_img_sample'+str(bi)+'.png', tar_img_polar[bi][0].clone().detach().cpu().numpy())
                            sif_tar_ng = sif_tar.clone().detach().requires_grad_(False)
                            one = torch.ones(sif_tar.shape).to(device)
                            zero = torch.zeros(sif_tar_ng.shape).to(device)
                            sif_tar_binary = torch.where(sif_tar_ng > 0, one, zero).requires_grad_(False)
                            tar_mask_polar_rep = torch.cat([tar_mask_polar]*7, dim=1).requires_grad_(False)
                            sif_shape = sif_out.shape
                            sif_out_masked = (nn.Sigmoid()(sif_out) * tar_mask_polar_rep).requires_grad_(True)
                            sif_tar_binary_masked = torch.clamp((sif_tar_binary * tar_mask_polar_rep).requires_grad_(False), 0, 1).float()
                            if args.sif_bce:
                                #sif_loss = nn.BCELoss()((nn.Flatten()(sif_out_masked)), nn.Flatten()(sif_tar_binary_masked))
                                sif_tar_masked = (nn.Sigmoid()(sif_tar_ng) * tar_mask_polar_rep).requires_grad_(False)
                                sif_loss = nn.BCELoss()(sif_out_masked, sif_tar_masked)
                            else:
                                #sif_loss = nn.SmoothL1Loss()(nn.Tanh()(sif_out), nn.Tanh()(sif_tar_ng))
                                sif_loss = nn.L1Loss()(sif_out_masked, sif_tar_masked)
                            if args.sif_gram:
                                sif_loss += nn.SmoothL1Loss()(sif_out_gram, sif_tar_gram)
                            epoch_sif_loss.append(args.gamma * sif_loss.item())
                            sif_out_binary = torch.where(sif_out_masked.clone().detach().requires_grad_(False) > 0.5, one, zero)
                            val_bit_diff.append(torch.mean(torch.abs(sif_out_binary * tar_mask_polar_rep - sif_tar_binary_masked).flatten()) * 100)
                        else:
                            sif_loss = 0.

                        loss = args.alpha * mse_loss + args.beta * mask_loss + args.gamma * sif_loss + args.epsilon * tv_loss + args.sigma * percept_loss
                        
                        val_epoch_loss.append(loss.item())
                        
                    val_loss_average += (sum(val_epoch_loss) / len(val_epoch_loss))
                    val_bit_diff_average += (sum(val_bit_diff) / len(val_bit_diff))
                
                val_loss_average /= args.val_repeats
                val_bit_diff_average /= args.val_repeats
                print("Val loss: {aver}, Val bit diff: {bit_diff} (epoch: {epoch})".format(aver = val_loss_average, bit_diff = val_bit_diff_average, epoch = epoch))
                if args.log_file is not None:
                    with open(args.log_file, 'a') as f:
                        f.write("Val loss: {aver}, Val bit diff: {bit_diff} (epoch: {epoch})".format(aver = val_loss_average, bit_diff = val_bit_diff_average, epoch = epoch) + '\n')
                
                if val_loss_average < best_val_loss:
                    best_val_loss = val_loss_average
                    if not os.path.exists(checkpoint_dir):
                        os.mkdir(checkpoint_dir)
                    filename = checkpoint_dir + "{epoch:04}-val_loss-{val_loss}-val_bit_diff-{bit_diff}.pth".format(epoch = epoch, use_mask_loss=args.use_mask_loss, use_sif_loss=args.use_sif_loss, val_loss=best_val_loss, bit_diff=val_bit_diff_average)
                    torch.save(conv_map_net, filename)
                    G_filename = checkpoint_dir + "{epoch:04}-val_loss-{val_loss}-val_bit_diff-{bit_diff}-G.pth".format(epoch = epoch, use_mask_loss=args.use_mask_loss, use_sif_loss=args.use_sif_loss, val_loss=best_val_loss, bit_diff=val_bit_diff_average)
                    torch.save(G, G_filename)
    return True
    
def evaluate(cfg, args):
    test_dataset = PairMinMaxBinDataset(args.test_bins_path, args.parent_dir, max_pairs_inp=4, input_size=(256,256))   
    test_dataloader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    epoch_num = int(args.weight_path.split('/')[-1].split('-')[0])
    
    if epoch_num < 10:
        print('Loading networks from "%s"...' % args.network_pkl)
        with dnnlib.util.open_url(args.network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
            G.eval()
    else:
        print('Loading trained generator ...')
        G_parent_dir = '/'.join(args.weight_path.split('/')[:-1])
        G_filename = '.'.join(args.weight_path.split('/')[-1].split('.')[:-1])+'-G.pth'
        G = torch.load(os.path.join(G_parent_dir, G_filename), map_location=device).to(device)
        G.eval()
    
        
    conv_map_net = torch.load(args.weight_path, map_location=device).to(device)
    conv_map_net.eval()
       
    code_rminp = []
    mask_rminp = []
    code_rmaxp = []
    mask_rmaxp = []
    code_pmaxp = []
    mask_pmaxp = []

    
    irisRec = irisRecognition(cfg, use_hough=False)
    
    args.val_repeats = 1
    
    with torch.no_grad():
        for rep in range(args.val_repeats):
            print('Repeat no:', str(rep+1)+'/'+str(args.val_repeats))
            if rep != 0:
                test_dataloader.dataset.reset()
            for batch, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        
                small_imgs = data['small_img']
                small_masks = data['small_mask']
                
                big_imgs = data['big_img']
                big_masks = data['big_mask']
                
                inp_mask = Variable(data['big_mask'])
                
                if args.cuda:
                    inp_mask.cuda()
                
                #Prepare input data
                small_img_pxyr = data['small_img_pxyr']
                small_img_ixyr = data['small_img_ixyr']
                
                big_img_pxyr = data['big_img_pxyr']
                big_img_ixyr = data['big_img_ixyr']
                
                #Prepare mapping input
                inp = Variable(torch.cat([small_imgs, big_masks], dim=1))
                
                if args.cuda:
                    inp = inp.cuda()
                
                #Find mapping to z
                z = conv_map_net(inp)
                
                #Generate image from z
                if hasattr(G.synthesis, 'input'):
                    m = make_transform(args.translate, args.rotate)
                    m = np.linalg.inv(m)
                    G.synthesis.input.transform.copy_(torch.from_numpy(m))
                label = torch.zeros([1, G.c_dim], device=device)
                out = torch.clamp(G(z, label, truncation_psi=1, noise_mode='const').clone().detach(), -1, 1)
                if batch == 0:
                    print(out.max(), small_imgs.max(), small_masks.max(), big_imgs.max(), big_masks.max())
                    print(out.min(), small_imgs.min(), small_masks.min(), big_imgs.min(), big_masks.min())
                
                #Unnormalize everything (resize output to 320x240)
                o_img_t = interpolate((out + 1)/2, size=(240, 320), mode='bilinear')
                small_img_t = interpolate((small_imgs + 1)/2, size=(240, 320), mode='bilinear')
                big_img_t = interpolate((big_imgs + 1)/2, size=(240, 320), mode='bilinear')
                small_mask_t = interpolate((small_masks + 1)/2, size=(240, 320), mode='nearest')
                big_mask_t = interpolate((big_masks + 1)/2, size=(240, 320), mode='nearest')
                
                
                #Find and append codes       
                for b in tqdm(range(small_img_t.shape[0]), total=small_img_t.shape[0]):
                    s_img = img_transform(small_img_t[b])
                    b_img = img_transform(big_img_t[b])
                    o_img = img_transform(o_img_t[b])
                    
                    s_mask = (small_mask_t[b][0].clone().detach().cpu().numpy()).astype(np.uint8)
                    b_mask = (big_mask_t[b][0].clone().detach().cpu().numpy()).astype(np.uint8)
                    
                    s_pxyr = np.around(small_img_pxyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                    s_ixyr = np.around(small_img_ixyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                    b_pxyr = np.around(big_img_pxyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                    b_ixyr = np.around(big_img_ixyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                    
                    s_img_polar, s_mask_polar = irisRec.cartToPol(s_img, s_mask, s_pxyr, s_ixyr)
                    b_img_polar, b_mask_polar = irisRec.cartToPol(b_img, b_mask, b_pxyr, b_ixyr)
                    
                    o_mask = irisRec.segment(o_img)
                    o_pxyr, o_ixyr = irisRec.circApprox(image=o_img)                  
                    o_img_polar, o_mask_polar = irisRec.cartToPol(o_img, o_mask, o_pxyr, o_ixyr)                    
                    
                    code_rminp.append(irisRec.extractCode(s_img_polar))
                    mask_rminp.append(s_mask_polar)
                    code_rmaxp.append(irisRec.extractCode(b_img_polar))
                    mask_rmaxp.append(b_mask_polar)
                    code_pmaxp.append(irisRec.extractCode(o_img_polar))
                    mask_pmaxp.append(o_mask_polar)
                    
                    if batch == 0:
                        all_polar_imgs = get_concat_v(get_concat_v(get_concat_v(Image.fromarray(s_img_polar.astype(np.uint8)), Image.fromarray(s_mask_polar.astype(np.uint8) * 255)), get_concat_v(Image.fromarray(b_img_polar.astype(np.uint8)), Image.fromarray(b_mask_polar.astype(np.uint8) * 255))), get_concat_v(Image.fromarray(o_img_polar.astype(np.uint8)), Image.fromarray(o_mask_polar.astype(np.uint8) * 255)))
                        
                        #code_shape = code_rminp[0].shape
                        #code1 = Image.fromarray(np.where(code_rminp[-1].reshape(code_shape[0]*code_shape[2], code_shape[1])>0, 255, 0).astype(np.uint8))
                        #code2 = Image.fromarray(np.where(code_rmaxp[-1].reshape(code_shape[0]*code_shape[2], code_shape[1])>0, 255, 0).astype(np.uint8))
                        #code3 = Image.fromarray(np.where(code_pmaxp[-1].reshape(code_shape[0]*code_shape[2], code_shape[1])>0, 255, 0).astype(np.uint8))
                        #all_codes = get_concat_v(get_concat_v(code1, code2), code3)
                        
                        all_polar_imgs.save('samples_eval/img_polar_sample_'+str(b)+'.png')
                        #all_codes.save('samples/code_sample_'+str(b)+'.png')
                        
                        s_img_c = s_img.convert('RGB')
                        b_img_c = b_img.convert('RGB')
                        o_img_c = o_img.convert('RGB')
                        
                        s_draw = ImageDraw.Draw(s_img_c)
                        s_draw.ellipse((s_pxyr[0]-s_pxyr[2], s_pxyr[1]-s_pxyr[2], s_pxyr[0]+s_pxyr[2], s_pxyr[1]+s_pxyr[2]), outline ="red")
                        s_draw.ellipse((s_ixyr[0]-s_ixyr[2], s_ixyr[1]-s_ixyr[2], s_ixyr[0]+s_ixyr[2], s_ixyr[1]+s_ixyr[2]), outline ="blue")
                        b_draw = ImageDraw.Draw(b_img_c)
                        b_draw.ellipse((b_pxyr[0]-b_pxyr[2], b_pxyr[1]-b_pxyr[2], b_pxyr[0]+b_pxyr[2], b_pxyr[1]+b_pxyr[2]), outline ="red")
                        b_draw.ellipse((b_ixyr[0]-b_ixyr[2], b_ixyr[1]-b_ixyr[2], b_ixyr[0]+b_ixyr[2], b_ixyr[1]+b_ixyr[2]), outline ="blue")
                        o_draw = ImageDraw.Draw(o_img_c)
                        o_draw.ellipse((o_pxyr[0]-o_pxyr[2], o_pxyr[1]-o_pxyr[2], o_pxyr[0]+o_pxyr[2], o_pxyr[1]+o_pxyr[2]), outline ="red")
                        o_draw.ellipse((o_ixyr[0]-o_ixyr[2], o_ixyr[1]-b_ixyr[2], o_ixyr[0]+b_ixyr[2], o_ixyr[1]+o_ixyr[2]), outline ="blue")
                        
                        all_imgs = get_concat_v(get_concat_v(get_concat_v(s_img_c, Image.fromarray(s_mask.astype(np.uint8) * 255)), get_concat_v(b_img_c, Image.fromarray(b_mask.astype(np.uint8) * 255))), get_concat_v(o_img_c, Image.fromarray(o_mask.astype(np.uint8) * 255)))
                        all_imgs.save('samples_eval/img_sample_'+str(b)+'.png')               
                
    
    inp_tar_scores_genuine = []
    out_tar_scores_genuine = []
    inp_tar_scores_imposter = []
    out_tar_scores_imposter = []
    fte = 0
    total = 0
    for i in tqdm(range(len(code_rminp)), total=len(code_rminp)):
        #print("Genuine: ")
        score_rminrmax, shift_rminrmax = irisRec.matchCodes(code_rminp[i], code_rmaxp[i], mask_rminp[i], mask_rmaxp[i])
        score_pmaxrmax, shift_pmaxrmax = irisRec.matchCodes(code_pmaxp[i], code_rmaxp[i], mask_pmaxp[i], mask_rmaxp[i])
        inp_tar_score = float(score_rminrmax)
        inp_tar_shift = float(shift_rminrmax)
        out_tar_score = float(score_pmaxrmax)
        out_tar_shift = float(shift_pmaxrmax)
        if out_tar_score < inp_tar_score:
            condition = "better"
        elif out_tar_score > inp_tar_score:
            condition = "worse"
        else:
            condition = "same"
        print(i, "Input <-> Target : {:.3f} (mutual rot: {:.1f} deg), Output <-> Target : {:.3f} (mutual rot: {:.1f} deg)".format(inp_tar_score,360*inp_tar_shift/512 ,out_tar_score,360*out_tar_shift/512), condition)
        with open('scores.txt', 'a') as f:
            f.write(str(i) + " Input <-> Target : {:.3f} (mutual rot: {:.1f} deg), Output <-> Target : {:.3f} (mutual rot: {:.1f} deg) ".format(inp_tar_score,360*inp_tar_shift/512 ,out_tar_score,360*out_tar_shift/512) + str(condition) + '\n')
        if inp_tar_score is not None and out_tar_score is not None and not math.isnan(inp_tar_score) and not math.isnan(out_tar_score):
            inp_tar_scores_genuine.append(inp_tar_score)
            out_tar_scores_genuine.append(out_tar_score)
            total += 1
        else:
            fte += 1
            total += 1
    
        for j in tqdm(range(i+1, len(code_rminp)), total=(len(code_rminp)-i-1)):

            score_rminrmax, shift_rminrmax = irisRec.matchCodes(code_rminp[i], code_rmaxp[j], mask_rminp[i], mask_rmaxp[j])
            score_pmaxrmax, shift_pmaxrmax = irisRec.matchCodes(code_pmaxp[i], code_rmaxp[j], mask_pmaxp[i], mask_rmaxp[j])
            inp_tar_score = float(score_rminrmax)
            inp_tar_shift = float(shift_rminrmax)
            out_tar_score = float(score_pmaxrmax)
            out_tar_shift = float(shift_pmaxrmax)
            
            if out_tar_score < inp_tar_score:
                condition = "better"
            elif out_tar_score > inp_tar_score:
                condition = "worse"
            else:
                condition = "same"
            if inp_tar_score is not None and out_tar_score is not None and not math.isnan(inp_tar_score) and not math.isnan(out_tar_score): 
                inp_tar_scores_imposter.append(inp_tar_score)
                out_tar_scores_imposter.append(out_tar_score)
                total += 1
            else:
                fte += 1
                total += 1
            
            print('\t', i, ' -> ', j, "Input ", i, " <-> Input ",  j, " : {:.3f} (mutual rot: {:.1f} deg), Output ".format(inp_tar_score,360*inp_tar_shift/512), i , " <-> Target ", j, " : {:.3f} (mutual rot: {:.1f} deg)".format(out_tar_score,360*out_tar_shift/512), condition)
            with open('scores.txt', 'a') as f:
                f.write('\t' + str(i) + ' -> ' + str(j) + " Input " + str(i) + " <-> Input " + str(j) + " : {:.3f} (mutual rot: {:.1f} deg), Output ".format(inp_tar_score,360*inp_tar_shift/512) + str(i) + " <-> Target " + str(j) + " : {:.3f} (mutual rot: {:.1f} deg)".format(out_tar_score,360*out_tar_shift/512) + str(condition) + '\n')

    params = {"ytick.color" : "b",
              "xtick.color" : "b",
              "axes.labelcolor" : "b",
              "axes.edgecolor" : "b"}
    plt.rcParams.update(params)

    save_dir = 'curves/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    fig5, axs5 = plt.subplots()
    itsg_counts, itsg_bins = np.histogram(inp_tar_scores_genuine, 5)
    otsg_counts, otsg_bins = np.histogram(out_tar_scores_genuine, 5)
    itsi_counts, itsi_bins = np.histogram(inp_tar_scores_imposter, 5)
    otsi_counts, otsi_bins = np.histogram(out_tar_scores_imposter, 5)
    itsg_probs = [count/itsg_counts.max() for count in itsg_counts]
    otsg_probs = [count/otsg_counts.max() for count in otsg_counts]
    itsi_probs = [count/itsi_counts.max() for count in itsi_counts]
    otsi_probs = [count/otsi_counts.max() for count in otsi_counts]

    axs5.plot(itsg_bins[:-1], itsg_probs, label='Input <-> Target (Genuine)')
    axs5.plot(otsg_bins[:-1], otsg_probs, label='Output <-> Target (Genuine)')
    axs5.plot(itsi_bins[:-1], itsi_probs, label='Input <-> Target (Imposter)')
    axs5.plot(otsi_bins[:-1], otsi_probs, label='Output <-> Target (Imposter)')
    fig5.legend(labelcolor='b')
    fig5.savefig(save_dir+'curve_histogram_5.png')
    
    fig6, axs6 = plt.subplots()
    itsg_counts, itsg_bins = np.histogram(inp_tar_scores_genuine, 10)
    otsg_counts, otsg_bins = np.histogram(out_tar_scores_genuine, 10)
    itsi_counts, itsi_bins = np.histogram(inp_tar_scores_imposter, 10)
    otsi_counts, otsi_bins = np.histogram(out_tar_scores_imposter, 10)
    itsg_probs = [count/itsg_counts.max() for count in itsg_counts]
    otsg_probs = [count/otsg_counts.max() for count in otsg_counts]
    itsi_probs = [count/itsi_counts.max() for count in itsi_counts]
    otsi_probs = [count/otsi_counts.max() for count in otsi_counts]

    axs6.plot(itsg_bins[:-1], itsg_probs, label='Input <-> Target (Genuine)')
    axs6.plot(otsg_bins[:-1], otsg_probs, label='Output <-> Target (Genuine)')
    axs6.plot(itsi_bins[:-1], itsi_probs, label='Input <-> Target (Imposter)')
    axs6.plot(otsi_bins[:-1], otsi_probs, label='Output <-> Target (Imposter)')
    fig6.legend(labelcolor='b')
    fig6.savefig(save_dir+'curve_histogram_10.png')

    fig11, axs11 = plt.subplots()
    itsg_counts, itsg_bins = np.histogram(inp_tar_scores_genuine, 15)
    otsg_counts, otsg_bins = np.histogram(out_tar_scores_genuine, 15)
    itsi_counts, itsi_bins = np.histogram(inp_tar_scores_imposter, 15)
    otsi_counts, otsi_bins = np.histogram(out_tar_scores_imposter, 15)
    itsg_probs = [count/itsg_counts.max() for count in itsg_counts]
    otsg_probs = [count/otsg_counts.max() for count in otsg_counts]
    itsi_probs = [count/itsi_counts.max() for count in itsi_counts]
    otsi_probs = [count/otsi_counts.max() for count in otsi_counts]
   
    axs11.plot(itsg_bins[:-1], itsg_probs, label='Input <-> Target (Genuine)')
    axs11.plot(otsg_bins[:-1], otsg_probs, label='Output <-> Target (Genuine)')
    axs11.plot(itsi_bins[:-1], itsi_probs, label='Input <-> Target (Imposter)')
    axs11.plot(otsi_bins[:-1], otsi_probs, label='Output <-> Target (Imposter)')
    fig11.legend(labelcolor='b')
    fig11.savefig(save_dir+'curve_histogram_15.png')
    
    fig12, axs12 = plt.subplots()
    itsg_counts, itsg_bins = np.histogram(inp_tar_scores_genuine, 20)
    otsg_counts, otsg_bins = np.histogram(out_tar_scores_genuine, 20)
    itsi_counts, itsi_bins = np.histogram(inp_tar_scores_imposter, 20)
    otsi_counts, otsi_bins = np.histogram(out_tar_scores_imposter, 20)
    itsg_probs = [count/itsg_counts.max() for count in itsg_counts]
    otsg_probs = [count/otsg_counts.max() for count in otsg_counts]
    itsi_probs = [count/itsi_counts.max() for count in itsi_counts]
    otsi_probs = [count/otsi_counts.max() for count in otsi_counts]

    axs12.plot(itsg_bins[:-1], itsg_probs, label='Input <-> Target (Genuine)')
    axs12.plot(otsg_bins[:-1], otsg_probs, label='Output <-> Target (Genuine)')
    axs12.plot(itsi_bins[:-1], itsi_probs, label='Input <-> Target (Imposter)')
    axs12.plot(otsi_bins[:-1], otsi_probs, label='Output <-> Target (Imposter)')
    fig12.legend(labelcolor='b')
    fig12.savefig(save_dir+'curve_histogram_20.png')
    
    itsg_mean = np.mean(inp_tar_scores_genuine)
    itsg_var = np.var(inp_tar_scores_genuine)
    itsi_mean = np.mean(inp_tar_scores_imposter)
    itsi_var = np.var(inp_tar_scores_imposter)
    its_d = abs(itsg_mean - itsi_mean)/math.sqrt(0.5 * (itsg_var + itsi_var))
    
    otsg_mean = np.mean(out_tar_scores_genuine)
    otsg_var = np.var(out_tar_scores_genuine)
    otsi_mean = np.mean(out_tar_scores_imposter)
    otsi_var = np.var(out_tar_scores_imposter)
    ots_d = abs(otsg_mean - otsi_mean)/math.sqrt(0.5 * (otsg_var + otsi_var))
    
    print('Non-Prednet d (Input <-> Target): ', its_d, ', Prednet d (Output <-> Target): ', ots_d, ', fte: ', fte/total)
    if args.log_file is not None:
        with open(args.log_file, 'a') as f:
            f.write('Non-Prednet d (Input <-> Target): ' + str(its_d) + ', Prednet d (Output <-> Target): ' + str(ots_d) + ', fte: '+ str(fte/total) + '\n')

def sample(args):
    
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst
    
    test_dataset = PairMinMaxBinDataset(args.test_bins_path, args.parent_dir, input_size=(256,256))
    test_dataloader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    epoch_num = int(args.weight_path.split('/')[-1].split('-')[0])
    
    if epoch_num < 10:
        print('Loading networks from "%s"...' % args.network_pkl)
        with dnnlib.util.open_url(args.network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    else:
        G_parent_dir = '/'.join(args.weight_path.split('/')[:-1])
        G_filename = '.'.join(args.weight_path.split('/')[-1].split('.')[:-1])+'-G.pth'
        print('Loading trained generator from', G_filename)
        G = torch.load(os.path.join(G_parent_dir, G_filename), map_location=device).to(device)
    
    G.eval()
    
    conv_map_net = torch.load(args.weight_path, map_location=device).to(device)
    conv_map_net.eval()    

    sample_path = './samples_' + '.'.join(args.weight_path.split('/')[-1].split('.')[:-1])
    
    maskLossModel = UNet(num_classes=2, num_channels=1).to(device)
    maskLossModel.load_state_dict(torch.load(args.mask_model_path, map_location=device))
    maskLossModel.eval()
        
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)
    
    with torch.no_grad():
        for batch, data in tqdm(enumerate(test_dataloader)):
        
            small_imgs = data['small_img']
            small_masks = data['small_mask']
            
            big_imgs = data['big_img']
            big_masks = data['big_mask']
            
            inp = Variable(torch.cat([small_imgs, big_masks], dim=1)).to(device)
            inp_mask = Variable(data['big_mask']).to(device)
            
            tar = Variable(big_imgs).to(device)
            z = conv_map_net(inp)
            
            if hasattr(G.synthesis, 'input'):
                m = make_transform(args.translate, args.rotate)
                m = np.linalg.inv(m)
                G.synthesis.input.transform.copy_(torch.from_numpy(m))
            
            label = torch.zeros([1, G.c_dim], device=device)
            out = G(z, label, truncation_psi=1, noise_mode='const')  
            out_norm = out - out.min()
            out_norm /= out_norm.max()
            mask_out_logprob = nn.LogSoftmax(dim=1)(maskLossModel(interpolate(out_norm, size=(240, 320), mode='bilinear')))
            mask_out = interpolate(torch.argmax(mask_out_logprob, dim=1).reshape(-1,1,240,320).float(), size=(256,256), mode='nearest')    
            
            for i in range(small_imgs.shape[0]):
                small_img_pil = (small_imgs[i].clone().detach().cpu().numpy() + 1)/2
                big_mask_pil = (big_masks[i].clone().detach().cpu().numpy() + 1)/2
                out_img_pil = np.clip(out_norm[i], 0, 1)
                big_img_pil = (big_imgs[i].clone().detach().cpu().numpy() + 1)/2
                img = np.concatenate([small_img_pil, big_mask_pil, out_img_pil, mask_out[i], big_img_pil], axis=1)
                img = np.moveaxis(img, 0, -1)
                img = np.uint8(img * 255)
                cv2.imwrite(sample_path + '/' + str(batch * args.batch_size + i) + '.png', img)
    return True 
                
    
def checkSIFLayer(cfg, args):
    irisRec = irisRecognition(cfg, use_hough=False)

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    test_dataset = PairMinMaxBinDataset(args.test_bins_path, args.parent_dir, max_pairs_inp=4, input_size=(240,320))   
    test_dataloader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    
    filter_mat = io.loadmat(args.sif_filter_path)['ICAtextureFilters']
    sifLossModel = SIFLayerMask(polar_height = 64, polar_width = 512, filter_mat = filter_mat, device=device).to(device)
    
    
    mean_sif_error = 0
    for batch, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        small_imgs = data['small_img']
        small_imgs_norm = (small_imgs + 1)/2
        
        big_imgs = data['big_img']
        big_imgs_norm = (big_imgs + 1)/2
        
        small_img_pxyr = torch.round(data['small_img_pxyr'])
        small_img_ixyr = torch.round(data['small_img_ixyr'])
        
        big_img_pxyr = torch.round(data['big_img_pxyr'])
        big_img_ixyr = torch.round(data['big_img_ixyr'])
        
        small_masks = data['small_mask']
        small_masks_norm = (small_masks + 1)/2
        #print('small_masks_norm max', small_masks_norm.max())
        
        big_masks = data['big_mask']
        big_masks_norm = (big_masks + 1)/2
        #print('big_masks_norm max', big_masks_norm.max())
        
        if args.cuda:
            small_imgs_norm = small_imgs_norm.cuda()
            big_imgs_norm = big_imgs_norm.cuda()
            small_masks_norm = small_masks_norm.cuda()
            big_masks_norm = big_masks_norm.cuda()
            #small_img_pxyr = small_img_pxyr.cuda()
            #small_img_ixyr = small_img_ixyr.cuda()
            #big_img_pxyr = big_img_pxyr.cuda()
            #big_img_ixyr = big_img_ixyr.cuda()
        
        sif_small_t, _, _ = sifLossModel(small_imgs_norm.clone().detach() * 255, small_img_pxyr, small_img_ixyr, mask=small_masks_norm)
        sif_big_t, _, _ = sifLossModel(big_imgs_norm.clone().detach() * 255, big_img_pxyr, big_img_ixyr, mask=big_masks_norm)
        
        small_mask_t = (small_masks + 1)/2
        big_mask_t = (big_masks + 1)/2
        
        sif_error = 0
        for b in tqdm(range(small_imgs_norm.shape[0]), total=small_imgs_norm.shape[0]):
            s_img = img_transform(small_imgs_norm[b])
            b_img = img_transform(big_imgs_norm[b])
            
            s_mask = (small_mask_t[b][0].clone().detach().cpu().numpy()).astype(np.uint8)
            b_mask = (big_mask_t[b][0].clone().detach().cpu().numpy()).astype(np.uint8)
            
            s_pxyr = np.around(small_img_pxyr[b].clone().detach().cpu().numpy()).astype(np.intc)
            s_ixyr = np.around(small_img_ixyr[b].clone().detach().cpu().numpy()).astype(np.intc)
            b_pxyr = np.around(big_img_pxyr[b].clone().detach().cpu().numpy()).astype(np.intc)
            b_ixyr = np.around(big_img_ixyr[b].clone().detach().cpu().numpy()).astype(np.intc)
            
            s_img_polar, s_mask_polar = irisRec.cartToPol(s_img, s_mask, s_pxyr, s_ixyr)
            b_img_polar, b_mask_polar = irisRec.cartToPol(b_img, b_mask, b_pxyr, b_ixyr)
            
            #print(s_mask)      
            
            small_sif_error_no = np.sum(np.logical_xor((sif_small_t[b].cpu().numpy()>0), np.moveaxis(irisRec.extractCode(s_img_polar), 2, 0)))
            big_sif_error_no = np.sum(np.logical_xor((sif_big_t[b].cpu().numpy()>0), np.moveaxis(irisRec.extractCode(b_img_polar), 2, 0)))          
            
            print('small sif error:', small_sif_error_no , '/', np.prod(sif_small_t[b].shape), '=', small_sif_error_no/np.prod(sif_small_t[b].shape) * 100)
            print('big sif error:', big_sif_error_no,'/',np.prod(sif_big_t[b].shape), '=', big_sif_error_no/np.prod(sif_big_t[b].shape) * 100 )
    #print('SIF error:', mean_sif_error)

        
    


                
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
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=10.0)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--log_batch', type=int, default=10)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--network_pkl', type=str, default='./models/network-snapshot-iris.pkl')
    parser.add_argument('--val_repeats', type=int, default=10)
    parser.add_argument('--train_gen', action='store_true')
    parser.add_argument('--use_corr_loss', action='store_true')
    parser.add_argument('--use_radial_corr_loss', action='store_true')
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--cfg_path', type=str, default="cfg.yaml", help="path of the iris recognition module configuration file.")
    parser.add_argument('--optim_type', type=str, default='adam')
    parser.add_argument('--virtual_batch_mult', type=int, default=4)
    parser.add_argument('--no_load_gen', action='store_true')
    parser.add_argument('--hinge_sif_loss', action='store_true')
    parser.add_argument('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0')
    parser.add_argument('--rotate', help='Rotation angle in degrees', type=float, default=0)
    parser.add_argument('--only_mapping', action='store_true')
    parser.add_argument('--sif_bce', action='store_true')
    parser.add_argument('--mask_bce', action='store_true')
    parser.add_argument('--sif_gram', action='store_true')
    parser.add_argument('--direct_bce', action='store_true')
    parser.add_argument('--train_min_max', action='store_true')
    parser.add_argument('--use_tv_loss', action='store_true')
    parser.add_argument('--use_perceptual_loss', action='store_true')
    
    args = parser.parse_args()
    
    #checkSIFLayer(get_cfg(args.cfg_path), args)
    
    #exit()      
    
    if args.evaluate:
        evaluate(get_cfg(args.cfg_path), args)
            
    if args.train:
        train(args)
    
    if args.sample:
        sample(args)
        
    if args.resume:
        resume(args)
    
    sys.stdout.close()
    
    
    
    
        
        
    