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
from torchvision.transforms import Normalize
from torch.nn.functional import interpolate

from dataset import PairFromBinDatasetSB, PairMinMaxBinDataset
from network import UNet, SIFLayerMask, TVLoss, VGGPerceptualLoss, DenseUNet, LPIPSLoss, L1LossWithSoftLabels, HingeLossWithSoftLabels, Discriminator, DenseUNetv2, NestedUNet, NestedUNetv2, HingeLoss
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
        self.loss = nn.L1Loss()
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
    checkpoint_dir = './dnet_both_cp_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + '/'
    softmax = nn.LogSoftmax(dim=1)
    # Declare Dataloaders
    train_dataset = PairFromBinDatasetSB(args.train_bins_path, args.parent_dir, res_mult=args.res_mult)
    train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    
    if args.val_bins_path:
        val_dataset = PairFromBinDatasetSB(args.val_bins_path, args.parent_dir, res_mult=args.res_mult)
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
        
    #conv_map_net = models.resnet50(pretrained=False)
    #conv_map_net.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #conv_map_net.fc = nn.Linear(2048, G.z_dim)
    #conv_map_net = conv_map_net.to(device)
    
    #conv_map_net = models.resnet18(pretrained=False)
    #conv_map_net.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #conv_map_net.fc = DenseLinearNetwork(n_layers=7, input_dim=512, hidden_dim=512, output_dim=G.z_dim)
    
    deform_net = torch.load(args.weight_path, map_location=device).to(device)
        
    print(deform_net)
    deform_net = deform_net.to(device)

    if args.use_tv_loss:
        tvLossModel = TVLoss(p=2)
    
    if args.use_vgg_loss:
        vggLossModel = VGGPerceptualLoss(device)
    
    if args.use_lpips_loss:
        lpipsLossModel = LPIPSLoss(device)

    if args.use_mask_loss:
        maskLossModel = UNet(num_classes=2, num_channels=1).to(device)
        maskLossModel.load_state_dict(torch.load(args.mask_model_path, map_location=device))
        maskLossModel.eval()
    
    if args.use_sif_loss:
        filter_mat = io.loadmat(args.sif_filter_path)['ICAtextureFilters']
        sifLossModel = SIFLayerMask(polar_height = 64, polar_width = 512, filter_mat = filter_mat, device=device).to(device)
        sifLossModel.eval()
    
    if args.optim_type == 'adam':
        optimizer = Adam(deform_net.parameters(), lr = args.lr)
    elif args.optim_type == 'sgd':
        optimizer = SGD(deform_net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim_type == 'cyclic_lr':
        optimizer = AdamW(deform_net.parameters(), lr=args.lr)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=10*args.lr, steps_per_epoch=len(train_dataloader), epochs=10)
    optimizer.zero_grad()
    #print(len(optimizer.param_groups))
    
    best_val_bit_diff = 100
    print('Starting Training...')
    sample_no = 0
    
    squared_hinge = HingeLoss(device, p=1)
    soft_hinge = HingeLossWithSoftLabels(device, label_smoothing=0.2)
    
    for epoch in range(1, args.num_epochs+1):
    
        deform_net.train()
        
        epoch_loss = []
        epoch_mse_loss = []
        epoch_mask_loss = []
        epoch_sif_loss = []
        epoch_tv_loss = []
        epoch_percept_loss = []
        epoch_lpips_loss = []
        epoch_sif_diff = []

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
            
            big_img_pxyr = data['big_img_pxyr']
            big_img_ixyr = data['big_img_ixyr']
            
            inp = Variable(torch.cat([small_imgs, big_masks], dim=1)).to(device)
            inp_mask = Variable(data['big_mask']).to(device, non_blocking=True)
            
            tar = Variable(big_imgs).to(device)

            out = deform_net(inp)
            
            out_norm = (out+1)/2
            tar_norm = (tar+1)/2                    
            
            mse_loss = nn.L1Loss()(out, tar.requires_grad_(False))
            epoch_mse_loss.append(args.alpha * mse_loss.item())
            
            if args.use_tv_loss:
                tv_loss = tvLossModel(out_norm)
                epoch_tv_loss.append(args.epsilon * tv_loss)
            else:
                tv_loss = 0.
            
            if args.use_vgg_loss:
                percept_loss = vggLossModel(out_norm, tar_norm.requires_grad_(False))
                epoch_percept_loss.append(args.sigma * percept_loss)
            else:
                percept_loss = 0.
            
            if args.use_lpips_loss:
                lpips_loss = torch.mean(lpipsLossModel(out, tar.requires_grad_(False)))
                epoch_lpips_loss.append(args.sigma * lpips_loss)
            else:
                lpips_loss = 0.
            
            mask_out_logprob = maskLossModel(interpolate(out_norm, size=(240, 320), mode='bilinear'))
            mask_inp_logprob = maskLossModel(interpolate(tar_norm, size=(240, 320), mode='bilinear')).detach().requires_grad_(False)
              
            if args.use_mask_loss:
                if args.mask_bce:
                    mask_loss = nn.BCELoss()(nn.Softmax(dim=1)(mask_out_logprob), nn.Softmax(dim=1)(mask_inp_logprob))
                else:
                    mask_loss = nn.L1Loss()(mask_out_logprob, mask_inp_logprob)
                epoch_mask_loss.append(args.beta * mask_loss.item())
            else:
                mask_loss = 0.
            
            if args.use_sif_loss:
                inp_mask_norm = ((inp_mask+1)/2)
                sif_out, sif_out_gram, out_img_polar, out_mask_polar = sifLossModel(out_norm, big_img_pxyr, big_img_ixyr, mask=inp_mask_norm)
                sif_tar, sif_tar_gram, tar_img_polar, tar_mask_polar = sifLossModel(tar_norm, big_img_pxyr, big_img_ixyr, mask=inp_mask_norm)
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
                minus_one = -1 * torch.ones(sif_tar.shape).to(device)
                sif_tar_hinge_masked = torch.where(sif_tar_binary_masked > 0.5, one, minus_one)
                sif_tar_hinge_inv_masked = torch.where(sif_tar_binary_masked < 0.5, one, minus_one)
                sif_out_hinge_masked = (nn.Tanh()(sif_out) * tar_mask_polar_rep).requires_grad_(True)
                sif_out_hinge_masked = torch.where(sif_out_hinge_masked == 0, minus_one, sif_out_hinge_masked)
                if args.sif_bce:
                    sif_out_sigmoid = nn.Sigmoid()(sif_out)
                    #sif_tar_bce_masked = torch.where(tar_mask_polar_rep == 0, sif_out_sigmoid, nn.Sigmoid()(sif_tar_ng))
                    sif_tar_bce_masked = torch.where(tar_mask_polar_rep == 0, sif_out_sigmoid, sif_tar_binary)
                    sif_loss = nn.BCEWithLogitsLoss()(nn.Flatten()(sif_out), nn.Flatten()(sif_tar_bce_masked))
                    #sif_loss = nn.BCELoss(reduction='sum')(sif_out_masked, sif_tar_masked) / torch.sum(tar_mask_polar_rep)   
                elif args.sif_hinge:
                    if args.sif_label_smoothing:
                        #sif_loss = 
                        sif_loss = soft_hinge(nn.Flatten()(sif_out_hinge_masked), nn.Flatten()(sif_tar_hinge_masked))
                    else:
                        #sif_loss = nn.SoftMarginLoss(nn.Flatten()(sif_out_hinge_masked), nn.Flatten()(sif_tar_hinge_masked))
                        ##sif_loss = 
                        #sif_loss = nn.HingeEmbeddingLoss()(nn.Flatten()(sif_out_hinge_masked), nn.Flatten()(sif_tar_hinge_inv_masked)) + 1
                        sif_loss = squared_hinge(nn.Flatten()(sif_out_hinge_masked), nn.Flatten()(sif_tar_hinge_masked))
                else:
                    #sif_loss = L1LossWithSoftLabels()(sif_out * tar_mask_polar_rep.float(), sif_tar_ng * tar_mask_polar_rep.float())
                    sif_tar_hinge_continuous_masked = (nn.Tanh()(sif_tar_ng) * tar_mask_polar_rep)
                    sif_loss = nn.L1Loss()(sif_out_hinge_masked, sif_tar_hinge_continuous_masked)
                epoch_sif_loss.append(args.gamma * sif_loss.item())
                sif_out_binary = torch.where(sif_out_masked.clone().detach().requires_grad_(False) > 0.5, one, zero)
                epoch_sif_diff.append(torch.mean(torch.abs(sif_out_binary * tar_mask_polar_rep - sif_tar_binary_masked).flatten()) * 100)
            else:
                sif_loss = 0.
            
            loss = args.alpha * mse_loss + args.beta * mask_loss + args.gamma * sif_loss + args.epsilon * tv_loss + args.sigma * (percept_loss + lpips_loss)
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
                optimizer.zero_grad()
            
            if vb_batch % args.log_batch == 0 and vb_count == 0:
                train_loss_average = sum(epoch_loss) / len(epoch_loss)
                mse_loss_average = sum(epoch_mse_loss) / len(epoch_mse_loss)
                loss_string = "Train loss: {aver} (epoch: {epoch}, batch: {batch}) Direct loss: {mse}".format(aver = train_loss_average, epoch = epoch, batch = vb_batch, mse = mse_loss_average)
                
                if args.use_tv_loss:
                    tv_loss_average = sum(epoch_tv_loss) / len(epoch_tv_loss)
                    loss_string += ", TV loss: {tv}".format(tv = tv_loss_average)
                
                if args.use_vgg_loss:
                    percept_loss_average = sum(epoch_percept_loss) / len(epoch_percept_loss)
                    loss_string += ", Perceptual loss: {percept}".format(percept = percept_loss_average) 
                
                if args.use_lpips_loss:
                    lpips_loss_average = sum(epoch_lpips_loss) / len(epoch_lpips_loss)
                    loss_string += ", LPIPS loss: {lpips}".format(lpips = lpips_loss_average)       
                
                if args.use_mask_loss:
                    mask_loss_average = sum(epoch_mask_loss) / len(epoch_mask_loss)
                    loss_string += ", Mask loss: {mask}".format(mask = mask_loss_average)
                
                if args.use_sif_loss:
                    sif_loss_average = sum(epoch_sif_loss) / len(epoch_sif_loss)
                    loss_string += ", SIF loss: {sif}".format(sif = sif_loss_average)
                    sif_bit_average = sum(epoch_sif_diff) / len(epoch_sif_diff)
                    loss_string += ", SIF bit diff: {sif_bit}%".format(sif_bit = sif_bit_average)
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
            deform_net.eval()
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
            
                        out = deform_net(inp)
                        
                        out_norm = (out+1)/2
                        tar_norm = (tar+1)/2
                        inp_mask_norm = ((inp_mask+1)/2)                    
                        
                        mse_loss = nn.L1Loss()(out, tar.requires_grad_(False))
                        
                        if args.use_tv_loss:
                            tv_loss = tvLossModel(out_norm)
                        else:
                            tv_loss = 0.
                        
                        if args.use_vgg_loss:
                            percept_loss = vggLossModel(out_norm, tar_norm.requires_grad_(False))
                        else:
                            percept_loss = 0.
                        
                        if args.use_lpips_loss:
                            lpips_loss = torch.mean(lpipsLossModel(out, tar.requires_grad_(False)))
                        else:
                            lpips_loss = 0.
                                    
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
                            sif_out, sif_out_gram, out_img_polar, out_mask_polar = sifLossModel(out_norm, big_img_pxyr, big_img_ixyr, mask=inp_mask_norm)
                            sif_tar, sif_tar_gram, tar_img_polar, tar_mask_polar = sifLossModel(tar_norm, big_img_pxyr, big_img_ixyr, mask=inp_mask_norm)
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
                            minus_one = -1 * torch.ones(sif_tar.shape).to(device)
                            sif_tar_hinge_masked = torch.where(sif_tar_binary_masked > 0.5, one, minus_one)
                            sif_tar_hinge_inv_masked =  torch.where(sif_tar_binary_masked < 0.5, one, minus_one)
                            sif_out_hinge_masked = (nn.Tanh()(sif_out) * tar_mask_polar_rep).requires_grad_(True)
                            sif_out_hinge_masked = torch.where(sif_out_hinge_masked == 0, minus_one, sif_out_hinge_masked)
                            if args.sif_bce:
                                sif_out_sigmoid = nn.Sigmoid()(sif_out)
                                #sif_tar_bce_masked = torch.where(tar_mask_polar_rep == 0, sif_out_sigmoid, nn.Sigmoid()(sif_tar_ng))
                                sif_tar_bce_masked = torch.where(tar_mask_polar_rep == 0, sif_out_sigmoid, sif_tar_binary)
                                sif_loss = nn.BCEWithLogitsLoss()(sif_out, sif_tar_bce_masked)
                                #sif_loss = nn.BCELoss(reduction='sum')(sif_out_masked, sif_tar_masked) / torch.sum(tar_mask_polar_rep)   
                            elif args.sif_hinge:
                                #sif_loss = nn.SoftMarginLoss(nn.Flatten()(sif_out_hinge_masked), nn.Flatten()(sif_tar_hinge_masked))
                                sif_loss = nn.HingeEmbeddingLoss()(nn.Flatten()(sif_out_hinge_masked), nn.Flatten()(sif_tar_hinge_inv_masked)) + 1
                                #sif_loss = squared_hinge(nn.Flatten()(sif_out_hinge_masked), nn.Flatten()(sif_tar_hinge_masked))
                            else:
                                #sif_loss = nn.L1Loss()(nn.Tanh()(sif_out), nn.Tanh()(sif_tar_ng))
                                sif_tar_hinge_continuous_masked = (nn.Tanh()(sif_tar_ng) * tar_mask_polar_rep)
                                sif_loss = nn.L1Loss()(sif_out_hinge_masked, sif_tar_hinge_continuous_masked)
                            epoch_sif_loss.append(args.gamma * sif_loss.item())
                            sif_out_binary = torch.where(sif_out_masked.clone().detach().requires_grad_(False) > 0.5, one, zero)
                            val_bit_diff.append(torch.mean(torch.abs(sif_out_binary * tar_mask_polar_rep - sif_tar_binary_masked).flatten()) * 100)
                        else:
                            sif_loss = 0.

                        loss = args.alpha * mse_loss + args.beta * mask_loss + args.gamma * sif_loss + args.epsilon * tv_loss + args.sigma * (percept_loss + lpips_loss)
                        
                        val_epoch_loss.append(loss.item())
                        
                    val_loss_average += (sum(val_epoch_loss) / len(val_epoch_loss))
                    val_bit_diff_average += (sum(val_bit_diff) / len(val_bit_diff))
                
                val_loss_average /= args.val_repeats
                val_bit_diff_average /= args.val_repeats
                print("Val loss: {aver}, Val bit diff: {bit_diff} (epoch: {epoch})".format(aver = val_loss_average, bit_diff = val_bit_diff_average, epoch = epoch))
                if args.log_file is not None:
                    with open(args.log_file, 'a') as f:
                        f.write("Val loss: {aver}, Val bit diff: {bit_diff} (epoch: {epoch})".format(aver = val_loss_average, bit_diff = val_bit_diff_average, epoch = epoch) + '\n')
                
                if val_bit_diff_average < best_val_bit_diff:
                    best_val_bit_diff = val_bit_diff_average
                    if not os.path.exists(checkpoint_dir):
                        os.mkdir(checkpoint_dir)
                    filename = checkpoint_dir + "{epoch:04}-val_bit_diff-{bit_diff}.pth".format(epoch = epoch, bit_diff=val_bit_diff_average)
                    torch.save(deform_net, filename)
    return True

def train(args):

    checkpoint_dir = './dnet_cp_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + '/'
    softmax = nn.LogSoftmax(dim=1)
    # Declare Dataloaders
    train_dataset = PairFromBinDatasetSB(args.train_bins_path, args.parent_dir, res_mult=args.res_mult) 
    train_dataloader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    
    if args.val_bins_path:
        val_dataset = PairFromBinDatasetSB(args.val_bins_path, args.parent_dir, res_mult=args.res_mult)
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
        
    #conv_map_net = models.resnet50(pretrained=False)
    #conv_map_net.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #conv_map_net.fc = nn.Linear(2048, G.z_dim)
    #conv_map_net = conv_map_net.to(device)
    
    #conv_map_net = models.resnet18(pretrained=False)
    #conv_map_net.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #conv_map_net.fc = DenseLinearNetwork(n_layers=7, input_dim=512, hidden_dim=512, output_dim=G.z_dim)
    
    if args.model_type == 'denseunet':
        deform_net = DenseUNet(num_classes=1, num_channels=2, width=32)
    if args.model_type == 'denseunetv2':
        deform_net = DenseUNetv2(num_classes=1, num_channels=2, width=32)
    elif args.model_type == 'nestedunet':
        deform_net = NestedUNet(num_classes=1, num_channels=2, width=32)
    elif args.model_type == 'resunet':
        deform_net = ResUNetv2(num_classes=1, num_channels=2, width=32)
    
    print(deform_net)    
        
    deform_net = deform_net.to(device)

    if args.use_tv_loss:
        tvLossModel = TVLoss(p=2)
    
    if args.use_vgg_loss:
        vggLossModel = VGGPerceptualLoss(device)
    
    if args.use_lpips_loss:
        lpipsLossModel = LPIPSLoss(device)
        
    if args.use_disc_loss:
        discLossModel = DiscPerceptLoss(args.network_pkl, device)

    if args.use_mask_loss:
        maskLossModel = UNet(num_classes=2, num_channels=1).to(device)
        maskLossModel.load_state_dict(torch.load(args.mask_model_path, map_location=device))
        maskLossModel.eval()
    
    if args.use_sif_loss:
        filter_mat = io.loadmat(args.sif_filter_path)['ICAtextureFilters']
        sifLossModel = SIFLayerMask(polar_height = 64, polar_width = 512, filter_mat = filter_mat, device=device).to(device)
        sifLossModel.eval()
    
    if args.optim_type == 'adam':
        optimizer = Adam(deform_net.parameters(), lr=args.lr, amsgrad=True)
    elif args.optim_type == 'sgd':
        optimizer = SGD(deform_net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim_type == 'cyclic_lr':
        optimizer = SGD(deform_net.parameters(), lr=args.lr, momentum=0.9)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=10*args.lr, steps_per_epoch=len(train_dataloader), epochs=10)
    optimizer.zero_grad()
    #print(len(optimizer.param_groups))
    
    best_val_bit_diff = 100
    print('Starting Training...')
    sample_no = 0
    
    squared_hinge = HingeLoss(device, p=2)
    soft_hinge = HingeLossWithSoftLabels(device, label_smoothing=0.2)
    
    for epoch in range(1, args.num_epochs+1):
    
        deform_net.train()
        
        epoch_loss = []
        epoch_mse_loss = []
        epoch_mask_loss = []
        epoch_sif_loss = []
        epoch_tv_loss = []
        epoch_percept_loss = []
        epoch_lpips_loss = []
        epoch_sif_diff = []
        epoch_disc_loss = []

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
            
            big_img_pxyr = data['big_img_pxyr']
            big_img_ixyr = data['big_img_ixyr']
            
            inp = Variable(torch.cat([small_imgs, big_masks], dim=1)).to(device)
            inp_mask = Variable(data['big_mask']).to(device, non_blocking=True)
            
            tar = Variable(big_imgs).to(device)

            out = deform_net(inp)
            
            out_norm = (out+1)/2
            tar_norm = (tar+1)/2                    
            
            mse_loss = nn.L1Loss()(out, tar.requires_grad_(False))
            epoch_mse_loss.append(args.alpha * mse_loss.item())
            
            if args.use_tv_loss:
                tv_loss = tvLossModel(out_norm)
                epoch_tv_loss.append(args.epsilon * tv_loss)
            else:
                tv_loss = 0.
            
            if args.use_vgg_loss:
                percept_loss = vggLossModel(out_norm, tar_norm.requires_grad_(False))
                epoch_percept_loss.append(args.sigma * percept_loss)
            else:
                percept_loss = 0.
            
            if args.use_lpips_loss:
                lpips_loss = torch.mean(lpipsLossModel(out, tar.requires_grad_(False)))
                epoch_lpips_loss.append(args.sigma * lpips_loss)
            else:
                lpips_loss = 0.
            
            if args.use_disc_loss:
                if args.disc_resize:
                    disc_loss = discLossModel(interpolate(out, size=(480,640), mode='bilinear'), interpolate(tar, size=(480,640), mode='bilinear').requires_grad_(False))
                else:
                    disc_loss = discLossModel(out, tar.requires_grad_(False))
                epoch_disc_loss.append(args.sigma * disc_loss)
            else:
                disc_loss = 0.
            
            mask_out_logprob = maskLossModel(interpolate(out_norm, size=(240, 320), mode='bilinear'))
            mask_inp_logprob = maskLossModel(interpolate(tar_norm, size=(240, 320), mode='bilinear')).detach().requires_grad_(False)
               
            if args.use_mask_loss:
                if args.mask_bce:
                    mask_loss = nn.BCELoss()(nn.Softmax(dim=1)(mask_out_logprob), nn.Softmax(dim=1)(mask_inp_logprob))
                else:
                    mask_loss = nn.L1Loss()(mask_out_logprob, mask_inp_logprob)
                epoch_mask_loss.append(args.beta * mask_loss.item())
            else:
                mask_loss = 0.
            
            if args.use_sif_loss:
                inp_mask_norm = ((inp_mask+1)/2)
                sif_out, sif_out_gram, out_img_polar, out_mask_polar = sifLossModel(out_norm, big_img_pxyr, big_img_ixyr, mask=inp_mask_norm)
                sif_tar, sif_tar_gram, tar_img_polar, tar_mask_polar = sifLossModel(tar_norm, big_img_pxyr, big_img_ixyr, mask=inp_mask_norm)
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
                minus_one = -1 * torch.ones(sif_tar.shape).to(device)
                sif_tar_hinge_masked = torch.where(sif_tar_binary_masked > 0.5, one, minus_one)
                sif_tar_hinge_inv_masked =  torch.where(sif_tar_binary_masked < 0.5, one, minus_one)
                sif_out_hinge_masked = (nn.Tanh()(sif_out) * tar_mask_polar_rep).requires_grad_(True)
                sif_out_hinge_masked = torch.where(sif_out_hinge_masked == 0, minus_one, sif_out_hinge_masked)
                if args.sif_bce:
                    sif_out_sigmoid = nn.Sigmoid()(sif_out)
                    #sif_tar_bce_masked = torch.where(tar_mask_polar_rep == 0, sif_out_sigmoid, nn.Sigmoid()(sif_tar_ng))
                    sif_tar_bce_masked = torch.where(tar_mask_polar_rep == 0, sif_out_sigmoid, sif_tar_binary)
                    sif_loss = nn.BCEWithLogitsLoss()(sif_out, sif_tar_bce_masked)
                    #sif_loss = nn.BCELoss(reduction='sum')(sif_out_masked, sif_tar_masked) / torch.sum(tar_mask_polar_rep)   
                elif args.sif_hinge:
                    if args.sif_label_smoothing:
                        sif_loss = soft_hinge(nn.Flatten()(sif_out_hinge_masked), nn.Flatten()(sif_tar_hinge_masked))
                    else:
                        sif_loss = squared_hinge(nn.Flatten()(sif_out_hinge_masked), nn.Flatten()(sif_tar_hinge_masked))
                        #sif_loss = nn.HingeEmbeddingLoss()(nn.Flatten()(sif_out_hinge_masked), nn.Flatten()(sif_tar_hinge_inv_masked)) + 1
                else:
                    #sif_loss = nn.L1Loss()(nn.Tanh()(sif_out), nn.Tanh()(sif_tar_ng))
                    #sif_loss = nn.L1Loss()(sif_out * tar_mask_polar_rep, sif_tar_ng * tar_mask_polar_rep)
                    sif_tar_hinge_continuous_masked = (nn.Tanh()(sif_tar_ng) * tar_mask_polar_rep)
                    sif_loss = nn.L1Loss()(sif_out_hinge_masked, sif_tar_hinge_continuous_masked)
                epoch_sif_loss.append(args.gamma * sif_loss.item())
                sif_out_binary = torch.where(sif_out_masked.clone().detach().requires_grad_(False) > 0.5, one, zero)
                epoch_sif_diff.append(torch.mean(torch.abs(sif_out_binary * tar_mask_polar_rep - sif_tar_binary_masked).flatten()) * 100)
            else:
                sif_loss = 0.
            
            loss = args.alpha * mse_loss + args.beta * mask_loss + args.gamma * sif_loss + args.epsilon * tv_loss + args.sigma * (percept_loss + lpips_loss + disc_loss)
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
                optimizer.zero_grad()
            
            if vb_batch % args.log_batch == 0 and vb_count == 0:
                train_loss_average = sum(epoch_loss) / len(epoch_loss)
                mse_loss_average = sum(epoch_mse_loss) / len(epoch_mse_loss)
                loss_string = "Train loss: {aver} (epoch: {epoch}, batch: {batch}) Direct loss: {mse}".format(aver = train_loss_average, epoch = epoch, batch = vb_batch, mse = mse_loss_average)
                
                if args.use_tv_loss:
                    tv_loss_average = sum(epoch_tv_loss) / len(epoch_tv_loss)
                    loss_string += ", TV loss: {tv}".format(tv = tv_loss_average)
                
                if args.use_vgg_loss:
                    percept_loss_average = sum(epoch_percept_loss) / len(epoch_percept_loss)
                    loss_string += ", Perceptual loss: {percept}".format(percept = percept_loss_average) 
                
                if args.use_lpips_loss:
                    lpips_loss_average = sum(epoch_lpips_loss) / len(epoch_lpips_loss)
                    loss_string += ", LPIPS loss: {lpips}".format(lpips = lpips_loss_average)  
                
                if args.use_disc_loss:
                    disc_loss_average = sum(epoch_disc_loss) / len(epoch_disc_loss)
                    loss_string += ", Disc loss: {disc}".format(disc = disc_loss_average)  
                         
                if args.use_mask_loss:
                    mask_loss_average = sum(epoch_mask_loss) / len(epoch_mask_loss)
                    loss_string += ", Mask loss: {mask}".format(mask = mask_loss_average)
                
                if args.use_sif_loss:
                    sif_loss_average = sum(epoch_sif_loss) / len(epoch_sif_loss)
                    loss_string += ", SIF loss: {sif}".format(sif = sif_loss_average)
                    sif_bit_average = sum(epoch_sif_diff) / len(epoch_sif_diff)
                    loss_string += ", SIF bit diff: {sif_bit}%".format(sif_bit = sif_bit_average)
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
            deform_net.eval()
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
            
                        out = deform_net(inp)
                        
                        out_norm = (out+1)/2
                        tar_norm = (tar+1)/2
                        inp_mask_norm = ((inp_mask+1)/2)                    
                        
                        mse_loss = nn.L1Loss()(out, tar.requires_grad_(False))
                        
                        if args.use_tv_loss:
                            tv_loss = tvLossModel(out_norm)
                        else:
                            tv_loss = 0.
                        
                        if args.use_vgg_loss:
                            percept_loss = vggLossModel(out_norm, tar_norm.requires_grad_(False))
                        else:
                            percept_loss = 0.
                        
                        if args.use_lpips_loss:
                            lpips_loss = torch.mean(lpipsLossModel(out, tar.requires_grad_(False)))
                        else:
                            lpips_loss = 0.
                        
                        if args.use_disc_loss:
                            if args.disc_resize:
                                disc_loss = discLossModel(interpolate(out, size=(480,640), mode='bilinear'), interpolate(tar, size=(480,640), mode='bilinear').requires_grad_(False))
                            else:
                                disc_loss = discLossModel(out, tar.requires_grad_(False))
                        else:
                            disc_loss = 0.
                                    
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
                            sif_out, sif_out_gram, out_img_polar, out_mask_polar = sifLossModel(out_norm, big_img_pxyr, big_img_ixyr, mask=inp_mask_norm)
                            sif_tar, sif_tar_gram, tar_img_polar, tar_mask_polar = sifLossModel(tar_norm, big_img_pxyr, big_img_ixyr, mask=inp_mask_norm)
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
                            minus_one = -1 * torch.ones(sif_tar.shape).to(device)
                            sif_tar_hinge_masked = torch.where(sif_tar_binary_masked > 0.5, one, minus_one)
                            sif_tar_hinge_inv_masked =  torch.where(sif_tar_binary_masked < 0.5, one, minus_one)
                            sif_out_hinge_masked = (nn.Tanh()(sif_out) * tar_mask_polar_rep).requires_grad_(True)
                            sif_out_hinge_masked = torch.where(sif_out_hinge_masked == 0, minus_one, sif_out_hinge_masked)
                            if args.sif_bce:
                                sif_out_sigmoid = nn.Sigmoid()(sif_out)
                                #sif_tar_bce_masked = torch.where(tar_mask_polar_rep == 0, sif_out_sigmoid, nn.Sigmoid()(sif_tar_ng))
                                sif_tar_bce_masked = torch.where(tar_mask_polar_rep == 0, sif_out_sigmoid, sif_tar_binary)
                                sif_loss = nn.BCEWithLogitsLoss()(sif_out, sif_tar_bce_masked)
                                #sif_loss = nn.BCELoss(reduction='sum')(sif_out_masked, sif_tar_masked) / torch.sum(tar_mask_polar_rep)   
                            elif args.sif_hinge:
                                sif_loss = squared_hinge(nn.Flatten()(sif_out_hinge_masked), nn.Flatten()(sif_tar_hinge_masked))
                                #sif_loss = nn.HingeEmbeddingLoss()(nn.Flatten()(sif_out_hinge_masked), nn.Flatten()(sif_tar_hinge_inv_masked)) + 1
                            else:
                                #sif_loss = nn.L1Loss()(nn.Tanh()(sif_out), nn.Tanh()(sif_tar_ng))
                                sif_tar_l1_masked = torch.where(sif_tar_binary_masked > 0.5, one, minus_one)
                                sif_loss = L1LossWithSoftLabels(device)(sif_out_hinge_masked, sif_tar_l1_masked)
                            epoch_sif_loss.append(args.gamma * sif_loss.item())
                            sif_out_binary = torch.where(sif_out_masked.clone().detach().requires_grad_(False) > 0.5, one, zero)
                            val_bit_diff.append(torch.mean(torch.abs(sif_out_binary * tar_mask_polar_rep - sif_tar_binary_masked).flatten()) * 100)
                        else:
                            sif_loss = 0.

                        loss = args.alpha * mse_loss + args.beta * mask_loss + args.gamma * sif_loss + args.epsilon * tv_loss + args.sigma * (percept_loss + lpips_loss + disc_loss)
                        
                        val_epoch_loss.append(loss.item())
                        
                    val_loss_average += (sum(val_epoch_loss) / len(val_epoch_loss))
                    val_bit_diff_average += (sum(val_bit_diff) / len(val_bit_diff))
                
                val_loss_average /= args.val_repeats
                val_bit_diff_average /= args.val_repeats
                print("Val loss: {aver}, Val bit diff: {bit_diff} (epoch: {epoch})".format(aver = val_loss_average, bit_diff = val_bit_diff_average, epoch = epoch))
                if args.log_file is not None:
                    with open(args.log_file, 'a') as f:
                        f.write("Val loss: {aver}, Val bit diff: {bit_diff} (epoch: {epoch})".format(aver = val_loss_average, bit_diff = val_bit_diff_average, epoch = epoch) + '\n')
                
                if val_bit_diff_average < best_val_bit_diff:
                    best_val_bit_diff = val_bit_diff_average
                    if not os.path.exists(checkpoint_dir):
                        os.mkdir(checkpoint_dir)
                    filename = checkpoint_dir + "{epoch:04}-val_bit_diff-{bit_diff}.pth".format(epoch = epoch, bit_diff=val_bit_diff_average)
                    torch.save(deform_net, filename)
    return True
    
def evaluate(cfg, args):
    test_dataset = PairMinMaxBinDataset(args.test_bins_path, args.parent_dir)   
    test_dataloader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')    
        
    deform_net = torch.load(args.weight_path, map_location=device).to(device)
    deform_net.eval()
       
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
                out = deform_net(inp)
                
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

def evaluate_biomech_minmax(cfg, args):
    test_dataset = PairMinMaxBinDataset(args.test_bins_path, args.parent_dir, max_pairs_inp = 5)   
    test_dataloader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')    
        
    if not os.path.exists('./polar_samples/'):
        os.mkdir('./polar_samples/')
    irisRec = irisRecognition(cfg, use_hough=False)
    
    deform_net = torch.load(args.weight_path, map_location=device).to(device)
    deform_net.eval()
    
    ds_ds = []
    bs_ds = []
    b2s_ds = []
    ns_ds = []
    
    bdgs_d = []
    bdgs_b = []
    bdgs_b2 = []
    bdgs_n = []
    
    bdis_d = []
    bdis_b = []
    bdis_b2 = []
    bdis_n = []
    
    with torch.no_grad():
        for rep in range(args.val_repeats):
            print('Repeat no:', str(rep+1)+'/'+str(args.val_repeats))
            if rep != 0:
                test_dataloader.dataset.reset()
            s_imgs = []
            b_imgs = []
            
            s_imgs_t = []
            b_masks_t = []
            
            s_pxyrs = []
            s_ixyrs = []
            b_pxyrs = []
            b_ixyrs = []
            
            s_masks = []
            b_masks = []
            
            pair_ids = []
            
            for batch, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        
                small_imgs = data['small_img']
                small_masks = data['small_mask']
                
                big_imgs = data['big_img']
                big_masks = data['big_mask']
                
                #Prepare input data
                small_img_pxyr = data['small_img_pxyr']
                small_img_ixyr = data['small_img_ixyr']
                
                big_img_pxyr = data['big_img_pxyr']
                big_img_ixyr = data['big_img_ixyr']
                
                identifiers = data['identifier']                
                
                #Unnormalize everything (resize output to 320x240)
                small_img_t = interpolate((small_imgs + 1)/2, size=(240, 320), mode='bilinear')
                big_img_t = interpolate((big_imgs + 1)/2, size=(240, 320), mode='bilinear')
                small_mask_t = interpolate((small_masks + 1)/2, size=(240, 320), mode='nearest')
                big_mask_t = interpolate((big_masks + 1)/2, size=(240, 320), mode='nearest')
                
                
                #Find and append codes       
                for b in tqdm(range(small_img_t.shape[0]), total=small_img_t.shape[0]):
                    s_img = img_transform(small_img_t[b])
                    b_img = img_transform(big_img_t[b])
                    
                    
                    s_mask = (small_mask_t[b][0].clone().detach().cpu().numpy()).astype(np.uint8)
                    b_mask = (big_mask_t[b][0].clone().detach().cpu().numpy()).astype(np.uint8)
                    
                    s_pxyr = np.around(small_img_pxyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                    s_ixyr = np.around(small_img_ixyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                    b_pxyr = np.around(big_img_pxyr[b].clone().detach().cpu().numpy()).astype(np.intc)
                    b_ixyr = np.around(big_img_ixyr[b].clone().detach().cpu().numpy()).astype(np.intc)                                  
                    
                    s_imgs.append(s_img)
                    b_imgs.append(b_img)
                    
                    s_imgs_t.append(small_imgs[b])
                    b_masks_t.append(big_masks[b])
                    
                    s_pxyrs.append(s_pxyr)
                    s_ixyrs.append(s_ixyr)
                    b_pxyrs.append(b_pxyr)
                    b_ixyrs.append(b_ixyr)
                    
                    s_masks.append(s_mask)
                    b_masks.append(b_mask)  
                    
                    pair_ids.append(identifiers[b])    
                
    
            daugman_scores_genuine = []
            biomech_scores_genuine = []
            biomech2_scores_genuine = []
            deformnet_scores_genuine = []
            daugman_scores_imposter = []
            biomech_scores_imposter = []
            biomech2_scores_imposter = []
            deformnet_scores_imposter = []
            fte = 0
            total = 0
            print('Finding scores...')
            bit_diff_daugman_genuine = []
            bit_diff_biomech_genuine = []
            bit_diff_biomech2_genuine = []
            bit_diff_deformnet_genuine = []
            bit_diff_daugman_imposter = []
            bit_diff_biomech_imposter = []
            bit_diff_biomech2_imposter = []
            bit_diff_deformnet_imposter = []
            for i in tqdm(range(len(s_imgs)), total=len(s_imgs)):

                s_im_polar, s_mask_polar = irisRec.cartToPol(s_imgs[i], s_masks[i], s_pxyrs[i], s_ixyrs[i])
                b_displacements = irisRec.findRadiusBiomech(s_pxyrs[i][2], b_pxyrs[i][2], s_ixyrs[i][2], b_ixyrs[i][2])
                b_im_polar, b_mask_polar = irisRec.bioMechCartToPol(b_imgs[i], b_masks[i], b_pxyrs[i], b_ixyrs[i], b_displacements)
                
                b_im_polar_2, b_mask_polar_2 = irisRec.cartToPol(b_imgs[i], b_masks[i], b_pxyrs[i], b_ixyrs[i])
                s_displacements = irisRec.findRadiusBiomech(b_pxyrs[i][2], s_pxyrs[i][2], b_ixyrs[i][2], s_ixyrs[i][2])
                s_im_polar_2, s_mask_polar_2 = irisRec.bioMechCartToPol(s_imgs[i], s_masks[i], s_pxyrs[i], s_ixyrs[i], s_displacements)

                s_im_polar_nb, s_mask_polar_nb = irisRec.cartToPol(s_imgs[i], s_masks[i], s_pxyrs[i], s_ixyrs[i])
                b_im_polar_nb, b_mask_polar_nb = irisRec.cartToPol(b_imgs[i], b_masks[i], b_pxyrs[i], b_ixyrs[i])
                
                inp = Variable(torch.cat([s_imgs_t[i], b_masks_t[i]], dim=0).unsqueeze(0)).to(device)
                
                out_img = deform_net(inp)
                o_img_t = interpolate((out_img + 1)/2, size=(240, 320), mode='bilinear')
                o_img = img_transform(o_img_t[0])
                
                o_im_polar, o_mask_polar = irisRec.cartToPol(o_img, b_masks[i], b_pxyrs[i], b_ixyrs[i])
                
                code_s_img_biomech = irisRec.extractCode(s_im_polar)
                code_b_img_biomech = irisRec.extractCode(b_im_polar)
                
                combined_biomech_mask = np.stack([s_mask_polar * b_mask_polar]*7, axis=2).astype(np.float32)
                combined_biomech_mask -= combined_biomech_mask.min()
                combined_biomech_mask /= combined_biomech_mask.max()
                combined_biomech_mask = np.where(combined_biomech_mask > 0.5, 1, 0)
                bit_diff_biomech_genuine.append(np.mean(np.absolute(code_s_img_biomech - code_b_img_biomech) * combined_biomech_mask) * 100)
                
                code_s_img_biomech2 = irisRec.extractCode(s_im_polar_2)
                code_b_img_biomech2 = irisRec.extractCode(b_im_polar_2)
                
                combined_biomech2_mask = np.stack([s_mask_polar_2 * b_mask_polar_2]*7, axis=2).astype(np.float32)
                combined_biomech2_mask -= combined_biomech2_mask.min()
                combined_biomech2_mask /= combined_biomech2_mask.max()
                combined_biomech2_mask = np.where(combined_biomech2_mask > 0.5, 1, 0)
                bit_diff_biomech2_genuine.append(np.mean(np.absolute(code_s_img_biomech2 - code_b_img_biomech2) * combined_biomech2_mask) * 100)
                
                code_s_img_nb = irisRec.extractCode(s_im_polar_nb)                
                code_b_img_nb = irisRec.extractCode(b_im_polar_nb)
                
                combined_daugman_mask = np.stack([s_mask_polar_nb * b_mask_polar_nb]*7, axis=2).astype(np.float32)
                combined_daugman_mask -= combined_daugman_mask.min()
                combined_daugman_mask /= combined_daugman_mask.max()
                combined_daugman_mask = np.where(combined_daugman_mask > 0.5, 1, 0)
                bit_diff_daugman_genuine.append(np.mean(np.absolute(code_s_img_nb - code_b_img_nb) * combined_daugman_mask) * 100)
                
                code_o_img = irisRec.extractCode(o_im_polar)
                
                deformnet_mask = np.stack([o_mask_polar]*7, axis=2).astype(np.float32)
                deformnet_mask -= deformnet_mask.min()
                deformnet_mask /= deformnet_mask.max()
                deformnet_mask = np.where(deformnet_mask > 0.5, 1, 0)
                bit_diff_deformnet_genuine.append(np.mean(np.absolute(code_o_img - code_b_img_nb) * deformnet_mask) * 100)
                
                score_daugman_genuine, shift_daugman_genuine = irisRec.matchCodes(code_s_img_nb, code_b_img_nb, s_mask_polar_nb, b_mask_polar_nb)
                score_biomech_genuine, shift_biomech_genuine = irisRec.matchCodes(code_s_img_biomech, code_b_img_biomech, s_mask_polar, b_mask_polar)
                score_biomech2_genuine, shift_biomech2_genuine = irisRec.matchCodes(code_s_img_biomech2, code_b_img_biomech2, s_mask_polar_2, b_mask_polar_2)
                score_deformnet_genuine, shift_deformnet_genuine = irisRec.matchCodes(code_o_img, code_b_img_nb, o_mask_polar, b_mask_polar_nb)
                
                if score_daugman_genuine is not None and score_biomech_genuine is not None and score_deformnet_genuine is not None and not math.isnan(score_daugman_genuine) and not math.isnan(score_biomech_genuine) and not math.isnan(score_deformnet_genuine): 
                    daugman_scores_genuine.append(score_daugman_genuine)
                    biomech_scores_genuine.append(score_biomech_genuine)
                    biomech2_scores_genuine.append(score_biomech2_genuine)
                    deformnet_scores_genuine.append(score_deformnet_genuine)
                    total += 1
                else:
                    fte += 1
                    total += 1
                
                id1 = pair_ids[i]
                
                for j in tqdm(range(i+1, len(s_imgs)), total=(len(s_imgs)-i-1)):
                    id2 = pair_ids[j]
                    s_im_polar, s_mask_polar = irisRec.cartToPol(s_imgs[i], s_masks[i], s_pxyrs[i], s_ixyrs[i])
                    b_displacements = irisRec.findRadiusBiomech(s_pxyrs[i][2], b_pxyrs[j][2], s_ixyrs[i][2], b_ixyrs[j][2])
                    b_im_polar, b_mask_polar = irisRec.bioMechCartToPol(b_imgs[j], b_masks[j], b_pxyrs[j], b_ixyrs[j], b_displacements)
                    
                    b_im_polar_2, b_mask_polar_2 = irisRec.cartToPol(b_imgs[j], b_masks[j], b_pxyrs[j], b_ixyrs[j])
                    s_displacements = irisRec.findRadiusBiomech(b_pxyrs[j][2], s_pxyrs[i][2], b_ixyrs[j][2], s_ixyrs[i][2])
                    s_im_polar_2, s_mask_polar_2 = irisRec.bioMechCartToPol(s_imgs[i], s_masks[i], s_pxyrs[i], s_ixyrs[i], s_displacements)

                    s_im_polar_nb, s_mask_polar_nb = irisRec.cartToPol(s_imgs[i], s_masks[i], s_pxyrs[i], s_ixyrs[i])
                    b_im_polar_nb, b_mask_polar_nb = irisRec.cartToPol(b_imgs[j], b_masks[j], b_pxyrs[j], b_ixyrs[j])
                    
                    inp = Variable(torch.cat([s_imgs_t[i], b_masks_t[j]], dim=0).unsqueeze(0)).to(device)
            
                    out_img = deform_net(inp)
                    o_img_t = interpolate((out_img + 1)/2, size=(240, 320), mode='bilinear')
                    o_img = img_transform(o_img_t[0])
                    
                    o_im_polar, o_mask_polar = irisRec.cartToPol(o_img, b_masks[j], b_pxyrs[j], b_ixyrs[j])
                    
                    code_s_img_biomech = irisRec.extractCode(s_im_polar)
                    code_b_img_biomech = irisRec.extractCode(b_im_polar)
                    
                    combined_biomech_mask = np.stack([s_mask_polar * b_mask_polar]*7, axis=2).astype(np.float32)
                    combined_biomech_mask -= combined_biomech_mask.min()
                    combined_biomech_mask /= combined_biomech_mask.max()
                    combined_biomech_mask = np.where(combined_biomech_mask > 0.5, 1, 0)
                    bit_diff_biomech = np.mean(np.absolute(code_s_img_biomech - code_b_img_biomech) * combined_biomech_mask) * 100
                    
                    code_s_img_biomech2 = irisRec.extractCode(s_im_polar_2)
                    code_b_img_biomech2 = irisRec.extractCode(b_im_polar_2)
                    
                    combined_biomech2_mask = np.stack([s_mask_polar_2 * b_mask_polar_2]*7, axis=2).astype(np.float32)
                    combined_biomech2_mask -= combined_biomech2_mask.min()
                    combined_biomech2_mask /= combined_biomech2_mask.max()
                    combined_biomech2_mask = np.where(combined_biomech2_mask > 0.5, 1, 0)
                    bit_diff_biomech2 = np.mean(np.absolute(code_s_img_biomech2 - code_b_img_biomech2) * combined_biomech2_mask) * 100
                    
                    code_s_img_nb = irisRec.extractCode(s_im_polar_nb)
                    code_b_img_nb = irisRec.extractCode(b_im_polar_nb)
                    
                    combined_daugman_mask = np.stack([s_mask_polar_nb * b_mask_polar_nb]*7, axis=2).astype(np.float32)
                    combined_daugman_mask -= combined_daugman_mask.min()
                    combined_daugman_mask /= combined_daugman_mask.max()
                    combined_daugman_mask = np.where(combined_daugman_mask > 0.5, 1, 0)
                    bit_diff_daugman = np.mean(np.absolute(code_s_img_nb - code_b_img_nb) * combined_daugman_mask) * 100
                    
                    code_o_img = irisRec.extractCode(o_im_polar)
                    
                    deformnet_mask = np.stack([o_mask_polar]*7, axis=2).astype(np.float32)
                    deformnet_mask -= deformnet_mask.min()
                    deformnet_mask /= deformnet_mask.max()
                    deformnet_mask = np.where(deformnet_mask > 0.5, 1, 0)
                    bit_diff_deformnet = np.mean(np.absolute(code_o_img - code_b_img_nb) * deformnet_mask) * 100
                    
                    score_daugman, shift_daugman = irisRec.matchCodes(code_s_img_nb, code_b_img_nb, s_mask_polar_nb, b_mask_polar_nb)
                    score_biomech, shift_biomech = irisRec.matchCodes(code_s_img_biomech, code_b_img_biomech, s_mask_polar, b_mask_polar)
                    score_biomech2, shift_biomech2 = irisRec.matchCodes(code_s_img_biomech2, code_b_img_biomech2, s_mask_polar_2, b_mask_polar_2)
                    score_deformnet, shift_deformnet = irisRec.matchCodes(code_o_img, code_b_img_nb, o_mask_polar, b_mask_polar_nb)
                    
                    if score_daugman is not None and score_biomech is not None and score_deformnet is not None and not math.isnan(score_daugman) and not math.isnan(score_biomech) and not math.isnan(score_deformnet):
                        if id1 != id2:                        
                            daugman_scores_imposter.append(score_daugman)
                            biomech_scores_imposter.append(score_biomech)
                            biomech2_scores_imposter.append(score_biomech2)
                            deformnet_scores_imposter.append(score_deformnet)
                            bit_diff_daugman_imposter.append(bit_diff_daugman)
                            bit_diff_biomech_imposter.append(bit_diff_biomech)
                            bit_diff_biomech2_imposter.append(bit_diff_biomech2)
                            bit_diff_deformnet_imposter.append(bit_diff_deformnet)
                        else:
                            daugman_scores_genuine.append(score_daugman)
                            biomech_scores_genuine.append(score_biomech)
                            biomech2_scores_genuine.append(score_biomech2)
                            deformnet_scores_genuine.append(score_deformnet)
                            bit_diff_daugman_genuine.append(bit_diff_daugman)
                            bit_diff_biomech_genuine.append(bit_diff_biomech)
                            bit_diff_biomech2_genuine.append(bit_diff_biomech2)
                            bit_diff_deformnet_genuine.append(bit_diff_deformnet)
                        total += 1
                    else:
                        fte += 1
                        total += 1
                        
            dsg_mean = np.mean(daugman_scores_genuine)
            dsg_var = np.var(daugman_scores_genuine)
            dsi_mean = np.mean(daugman_scores_imposter)
            dsi_var = np.var(daugman_scores_imposter)
            ds_d = abs(dsg_mean - dsi_mean)/math.sqrt(0.5 * (dsg_var + dsi_var))
            ds_ds.append(ds_d)                
            
            bsg_mean = np.mean(biomech_scores_genuine)
            bsg_var = np.var(biomech_scores_genuine)
            bsi_mean = np.mean(biomech_scores_imposter)
            bsi_var = np.var(biomech_scores_imposter)
            bs_d = abs(bsg_mean - bsi_mean)/math.sqrt(0.5 * (bsg_var + bsi_var))
            bs_ds.append(bs_d)
            
            b2sg_mean = np.mean(biomech2_scores_genuine)
            b2sg_var = np.var(biomech2_scores_genuine)
            b2si_mean = np.mean(biomech2_scores_imposter)
            b2si_var = np.var(biomech2_scores_imposter)
            b2s_d = abs(b2sg_mean - b2si_mean)/math.sqrt(0.5 * (b2sg_var + b2si_var))
            b2s_ds.append(b2s_d)
            
            nsg_mean = np.mean(deformnet_scores_genuine)
            nsg_var = np.var(deformnet_scores_genuine)
            nsi_mean = np.mean(deformnet_scores_imposter)
            nsi_var = np.var(deformnet_scores_imposter)
            ns_d = abs(nsg_mean - nsi_mean)/math.sqrt(0.5 * (nsg_var + nsi_var))
            ns_ds.append(ns_d)
            
            bdg_d = sum(bit_diff_daugman_genuine)/len(bit_diff_daugman_genuine)
            bdgs_d.append(bdg_d)
            bdg_b = sum(bit_diff_biomech_genuine)/len(bit_diff_biomech_genuine)
            bdgs_b.append(bdg_b)
            bdg_b2 = sum(bit_diff_biomech2_genuine)/len(bit_diff_biomech2_genuine)
            bdgs_b2.append(bdg_b2)
            bdg_n = sum(bit_diff_deformnet_genuine)/len(bit_diff_deformnet_genuine)
            bdgs_n.append(bdg_n)
            
            bdi_d = sum(bit_diff_daugman_imposter)/len(bit_diff_daugman_imposter)
            bdis_d.append(bdi_d)
            bdi_b = sum(bit_diff_biomech_imposter)/len(bit_diff_biomech_imposter)
            bdis_b.append(bdi_b)
            bdi_b2 = sum(bit_diff_biomech2_imposter)/len(bit_diff_biomech2_imposter)
            bdis_b2.append(bdi_b2)
            bdi_n = sum(bit_diff_deformnet_imposter)/len(bit_diff_deformnet_imposter)
            bdis_n.append(bdi_n)
            
            print('Bit difference genuine Daugman:', bdg_d, ', Bit difference genuine Biomech:', bdg_b, ', Bit difference genuine Biomech2:', bdg_b2, ', Bit difference genuine deformnet:', bdg_n)
            print('Bit difference imposter Daugman:', bdi_d, ', Bit difference imposter Biomech:', bdi_b, ', Bit difference imposter Biomech2:', bdi_b2, ', Bit difference imposter deformnet:', bdi_n)
            print('Daugman d\' (Input <-> Target): ', ds_d, ', Biomech d\' (Output <-> Target): ', bs_d, ', Biomech2 d\' (Output <-> Target): ', b2s_d,  ', Deformnet d\' (Output <-> Target): ', ns_d, ', fte: ', fte/total)   
    
    print('Bit difference genuine Daugman:', np.mean(bdgs_d), '+-', np.std(bdgs_d), ', Bit difference genuine Biomech:', np.mean(bdgs_b), '+-', np.std(bdgs_b), ', Bit difference genuine Biomech2:', np.mean(bdgs_b2), '+-', np.std(bdgs_b2), ', Bit difference genuine deformnet:', np.mean(bdgs_n), '+-', np.std(bdgs_n))
    print('Bit difference imposter Daugman:', np.mean(bdis_d), '+-', np.std(bdis_d), ', Bit difference imposter Biomech:', np.mean(bdis_b), '+-', np.std(bdis_b), ', Bit difference imposter Biomech2:', np.mean(bdis_b2), '+-', np.std(bdis_b2), ', Bit difference imposter deformnet:', np.mean(bdis_n), '+-', np.std(bdis_n))
    print('Daugman d\' (Input <-> Target): ', np.mean(ds_ds), '+-', np.std(ds_ds), ', Biomech d\' (Output <-> Target): ', np.mean(bs_ds), '+-', np.std(bs_ds), ', Biomech2 d\' (Output <-> Target): ', np.mean(b2s_ds), '+-', np.std(b2s_ds), ', Deformnet d\' (Output <-> Target): ', np.mean(ns_ds), '+-', np.std(ns_ds), ', fte: ', fte/total)   

def sample(args):
    
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst
    
    test_dataset = PairMinMaxBinDataset(args.test_bins_path, args.parent_dir, max_pairs_inp=2)
    test_dataloader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    deform_net = torch.load(args.weight_path, map_location=device).to(device)
    deform_net.eval()    

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
            out = deform_net(inp)
            
            out_norm = out - out.min()
            out_norm /= out_norm.max()
            mask_out_logprob = nn.LogSoftmax(dim=1)(maskLossModel(interpolate(out_norm, size=(240, 320), mode='bilinear')))
            mask_out = torch.argmax(mask_out_logprob, dim=1).reshape(-1,1,240,320).float() 
            
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

def makevideo(args, input_size=(240,320)):
    test_video_keys = PairMinMaxBinDataset(args.test_bins_path, args.parent_dir).bins.keys()
    
    for vid_id in test_video_keys:
        print('Creating video for', vid_id)
        if not os.path.exists('video_both_'+vid_id):
            os.mkdir('video_both_'+vid_id)
        with torch.no_grad():
            device = torch.device('cpu')
                
            transform = transforms.Compose([
                    transforms.Resize([input_size[0], input_size[1]]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, ), (0.5, ))
            ])
            
            deform_net = torch.load(args.weight_path, map_location=device).to(device)
            deform_net.eval()
            
            maskModel = UNet(num_classes=2, num_channels=1).to(device)
            maskModel.load_state_dict(torch.load(args.mask_model_path, map_location=device))
            maskModel.eval()
            
            video_folder = os.path.join(args.vid_src, vid_id.split('_')[0])
            orig_video = {}
            orig_video_mask = {}
            gen_video = {}
            gen_video_mask = {}
            print('Loading original video...')
            for filename in tqdm(os.listdir(video_folder)):
                if filename.endswith(('jpg','jpeg','bmp','gif','png','tiff')):
                    if filename.startswith(vid_id):
                        filepath = os.path.join(video_folder, filename)
                        frame_num = int(filename.split('.')[0].split('_')[-1])
                        frame = transform(Image.open(filepath).convert('L'))
                        if frame_num == 1:
                            orig_video_first_frame = frame
                        frame_norm = (frame+1)/2
                        mask_logprob = maskModel(frame_norm.unsqueeze(0))
                        mask = torch.argmax(mask_logprob, dim=1).reshape(-1,1,240,320)
                        orig_video[frame_num] = frame
                        orig_video_mask[frame_num] = (mask - 0.5)/0.5
            
            print('Generating video...')
            min_frame_num = min(orig_video.keys())
            first_frame = ((orig_video[min_frame_num]+1)/2).unsqueeze(0)
            print('Minimum frame number:', min_frame_num)
            for frame_num in tqdm(sorted(orig_video.keys()), total=len(orig_video.keys())):
                inp = Variable(torch.cat([first_frame, orig_video_mask[frame_num]], dim=1)).to(device)
                out = deform_net(inp)
                out_norm = out - out.min()
                out_norm /= out_norm.max()
                mask_logprob = maskModel(out_norm)
                mask = torch.argmax(mask_logprob, dim=1).reshape(-1,1,240,320)
                orig_frame = ((orig_video[frame_num]+1)/2)[0].cpu().numpy()
                orig_mask = ((orig_video_mask[frame_num]+1)/2)[0][0].cpu().numpy()
                gen_frame = out_norm[0][0].detach().cpu().numpy()
                gen_mask = mask[0][0].detach().cpu().numpy()
                input_frame = first_frame[0][0].detach().cpu().numpy()
                #orig_frame = np.concatenate([orig_frame.reshape(input_size[0], input_size[1], 1)]*3, axis=2)
                #orig_mask = np.concatenate([orig_mask.reshape(input_size[0], input_size[1], 1)]*3, axis=2)
                #gen_frame = np.concatenate([gen_frame.reshape(input_size[0], input_size[1], 1)]*3, axis=2)
                #gen_mask = np.concatenate([gen_mask.reshape(input_size[0], input_size[1], 1)]*3, axis=2)
                #first_frame = np.concatenate([first_frame.reshape(input_size[0], input_size[1], 1)]*3, axis=2)
                
                frame_part1 = np.concatenate([orig_frame, gen_frame], axis=1)
                frame_part2 = np.concatenate([orig_mask, input_frame], axis=1)
                frame = np.concatenate([frame_part1, frame_part2], axis=0)
                
                cv2.imwrite('video_both_'+vid_id+'/image-'+str(frame_num)+'.png', np.uint8(frame * 255))       
    


                
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
    parser.add_argument('--gamma', type=float, default=1.6)
    parser.add_argument('--epsilon', type=float, default=0.0)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--log_batch', type=int, default=1)
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
    
    #checkSIFLayer(get_cfg(args.cfg_path), args)
    
    #exit()      
            
    if args.train:
        train(args)
    elif args.resume:
        resume(args)
    elif args.evaluate:
        evaluate(get_cfg(args.cfg_path), args)
    elif args.evaluate_minmax:
        evaluate_biomech_minmax(get_cfg(args.cfg_path), args)
    elif args.sample:
        sample(args)  
    elif args.make_video:
        makevideo(args)
    
    sys.stdout.close()
    cv2.destroyAllWindows()