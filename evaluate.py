import numpy as np
import torch
import os
import math

import cv2
from argparse import ArgumentParser
import torch.nn as nn


from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from torch.nn.functional import interpolate

from dataset import PairFromBinDatasetSB, PairMinMaxBinDataset

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
torch.backends.cudnn.deterministic = True

def img_transform(img_tensor):
    img_t = img_tensor[0]
    img_t = np.clip(img_t.clone().detach().cpu().numpy() * 255, 0, 255)
    img = Image.fromarray(img_t.astype(np.uint8))
    return img

def evaluate(cfg, args):
    test_dataset = PairFromBinDatasetSB(args.test_bins_path, args.parent_dir)   
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


def evaluate_biomech_minmax(cfg, args):
    test_dataset = PairMinMaxBinDataset(args.test_bins_path, args.parent_dir, max_pairs_inp = 2)   
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
    parser.add_argument('--mask_model_path', type=str, default='./models/CCNet_epoch_260_NIRRGBmixed_adam.pth')
    parser.add_argument('--sif_filter_path', type=str, default='./models/ICAtextureFilters_15x15_7bit.mat')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--evaluate_minmax', action='store_true')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--val_repeats', type=int, default=10)
    parser.add_argument('--cfg_path', type=str, default="cfg.yaml", help="path of the iris recognition module configuration file.")
    parser.add_argument('--make_video', action='store_true')
    parser.add_argument('--vid_src', type=str, default='/data1/warsaw_pupil_dynamics/WarsawData/')
    parser.add_argument('--res_mult', type=int, default=1)

    
    args = parser.parse_args()
        
    if args.evaluate_minmax:
        evaluate_biomech_minmax(get_cfg(args.cfg_path), args)
    elif args.sample:
        sample(args)  
    elif args.make_video:
        makevideo(args)
    else:
        evaluate(get_cfg(args.cfg_path), args)
