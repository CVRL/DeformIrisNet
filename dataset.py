import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from PIL import Image
import random
import os
import pickle as pkl
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import models, transforms
from math import pi
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

class AllPairsDatasetSB(Dataset):
    def __init__(self, bins_path, parent_dir, flip_data = True, res_mult = 1):
        super().__init__()
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.pupil_xyrs_path = os.path.join(parent_dir, 'WarsawPupilCircles.pkl')
        self.iris_xyrs_path = os.path.join(parent_dir, 'WarsawIrisCircles.pkl') 
        self.flip_data = flip_data
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
        
        
        print('Finding all pairs...')
        self.pairs = []
        self.pair_identifiers = []
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        for img1 in self.bins[identifier][bin_num_1]:
                            for img2 in self.bins[identifier][bin_num_2]:
                                self.pairs.append([img1, img2])   
                                self.pair_identifiers.append(identifier)
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        with open(self.pupil_xyrs_path, 'rb') as pupilFile:
            self.pupil_xyrs = pkl.load(pupilFile)
        
        with open(self.iris_xyrs_path, 'rb') as irisFile:
            self.iris_xyrs = pkl.load(irisFile)

    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        small_img_name = pair[0].strip()
        big_img_name = pair[1].strip()
        
        indiv_id1 = small_img_name.split('_')[0]
        indiv_id2 = big_img_name.split('_')[0]
        
        small_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), small_img_name)
        small_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id1), small_img_name)
        
        big_img_path = os.path.join(os.path.join(self.image_dir, indiv_id2), big_img_name)
        big_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), big_img_name)
        
        small_img = self.load_image(small_img_path)
        small_mask = self.load_image(small_mask_path)
        
        big_img = self.load_image(big_img_path)
        big_mask = self.load_image(big_mask_path)
        
        small_img_pupil_xyr = torch.tensor(self.pupil_xyrs[small_img_name]) * self.res_mult
        small_img_iris_xyr = torch.tensor(self.iris_xyrs[small_img_name]) * self.res_mult
        
        big_img_pupil_xyr = torch.tensor(self.pupil_xyrs[big_img_name]) * self.res_mult
        big_img_iris_xyr = torch.tensor(self.iris_xyrs[big_img_name]) * self.res_mult
        
        if self.flip_data and random.random() < 0.5:
            small_img = small_img.transpose(Image.FLIP_LEFT_RIGHT) 
            small_mask = small_mask.transpose(Image.FLIP_LEFT_RIGHT)
            big_img = big_img.transpose(Image.FLIP_LEFT_RIGHT) 
            big_mask = big_mask.transpose(Image.FLIP_LEFT_RIGHT)
            small_img_pupil_xyr[0] = self.input_size[1] - small_img_pupil_xyr[0]
            big_img_pupil_xyr[0] = self.input_size[1] - big_img_pupil_xyr[0]
            small_img_iris_xyr[0] = self.input_size[1] - small_img_iris_xyr[0]
            big_img_iris_xyr[0] = self.input_size[1] - big_img_iris_xyr[0]
            
            
        xform_small_img = self.transform(small_img)
        xform_small_mask = self.transform(small_mask)
        xform_small_mask[xform_small_mask<0] = -1
        xform_small_mask[xform_small_mask>=0] = 1
        
        xform_big_img = self.transform(big_img)
        xform_big_mask = self.transform(big_mask)
        xform_big_mask[xform_big_mask<0] = -1
        xform_big_mask[xform_big_mask>=0] = 1
        
        return {'small_img': xform_small_img, 'small_mask' : xform_small_mask, 'big_img' : xform_big_img, 'big_mask' : xform_big_mask, 'small_img_pxyr': small_img_pupil_xyr, 'small_img_ixyr' : small_img_iris_xyr, 'big_img_pxyr' : big_img_pupil_xyr, 'big_img_ixyr' :  big_img_iris_xyr }
    
    def __len__(self):
        return len(self.pairs)

class PairMinMaxBinDataset(Dataset):
    def __init__(self, bins_path, parent_dir, max_pairs_inp = None, res_mult = 1):
        super().__init__()
        
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.pupil_xyrs_path = os.path.join(parent_dir, 'WarsawPupilCircles.pkl')
        self.iris_xyrs_path = os.path.join(parent_dir, 'WarsawIrisCircles.pkl')
        self.max_pairs_inp = max_pairs_inp
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
           
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
        
        print('Setting Dataset for Min-max pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][min_bin_num][img_ind], self.bins[identifier][max_bin_num][img_ind]])
                self.pair_ids.append('_'.join(identifier.split('_')[:-1]))
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        with open(self.pupil_xyrs_path, 'rb') as pupilFile:
            self.pupil_xyrs = pkl.load(pupilFile)
        
        with open(self.iris_xyrs_path, 'rb') as irisFile:
            self.iris_xyrs = pkl.load(irisFile)

    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        small_img_name = pair[0].strip()
        big_img_name = pair[1].strip()
        
        indiv_id1 = small_img_name.split('_')[0]
        indiv_id2 = big_img_name.split('_')[0]
        
        small_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), small_img_name)
        small_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id1), small_img_name)
        
        big_img_path = os.path.join(os.path.join(self.image_dir, indiv_id2), big_img_name)
        big_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), big_img_name)
        
        small_img = self.load_image(small_img_path)
        small_mask = self.load_image(small_mask_path)
        
        big_img = self.load_image(big_img_path)
        big_mask = self.load_image(big_mask_path)
        
        small_img_pupil_xyr = torch.tensor(self.pupil_xyrs[small_img_name]) * self.res_mult
        small_img_iris_xyr = torch.tensor(self.iris_xyrs[small_img_name]) * self.res_mult
        
        big_img_pupil_xyr = torch.tensor(self.pupil_xyrs[big_img_name]) * self.res_mult
        big_img_iris_xyr = torch.tensor(self.iris_xyrs[big_img_name]) * self.res_mult
        
        xform_small_img = self.transform(small_img)
        xform_small_mask = self.transform(small_mask)
        xform_small_mask[xform_small_mask<0] = -1
        xform_small_mask[xform_small_mask>=0] = 1
        
        xform_big_img = self.transform(big_img)
        xform_big_mask = self.transform(big_mask)
        xform_big_mask[xform_big_mask<0] = -1
        xform_big_mask[xform_big_mask>=0] = 1
        
        return {'small_img': xform_small_img, 'small_mask' : xform_small_mask, 'big_img' : xform_big_img, 'big_mask' : xform_big_mask, 'small_img_pxyr': small_img_pupil_xyr, 'small_img_ixyr' : small_img_iris_xyr, 'big_img_pxyr' : big_img_pupil_xyr, 'big_img_ixyr' :  big_img_iris_xyr, 'identifier':self.pair_ids[index]}
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Setting Dataset for Min-max pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][min_bin_num][img_ind], self.bins[identifier][max_bin_num][img_ind]])
                self.pair_ids.append('_'.join(identifier.split('_')[:-1]))
                
class PairMaxMinBinDataset(Dataset):
    def __init__(self, bins_path, parent_dir, max_pairs_inp = None, res_mult=1):
        super().__init__()
        
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.pupil_xyrs_path = os.path.join(parent_dir, 'WarsawPupilCircles.pkl')
        self.iris_xyrs_path = os.path.join(parent_dir, 'WarsawIrisCircles.pkl')
        self.max_pairs_inp = max_pairs_inp
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
        
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
        
        print('Setting Dataset for Min-max pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][max_bin_num][img_ind], self.bins[identifier][min_bin_num][img_ind]])
                self.pair_ids.append('_'.join(identifier.split('_')[:-1])) 
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        with open(self.pupil_xyrs_path, 'rb') as pupilFile:
            self.pupil_xyrs = pkl.load(pupilFile)
        
        with open(self.iris_xyrs_path, 'rb') as irisFile:
            self.iris_xyrs = pkl.load(irisFile)

    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        small_img_name = pair[0].strip()
        big_img_name = pair[1].strip()
        
        indiv_id1 = small_img_name.split('_')[0]
        indiv_id2 = big_img_name.split('_')[0]
        
        small_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), small_img_name)
        small_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id1), small_img_name)
        
        big_img_path = os.path.join(os.path.join(self.image_dir, indiv_id2), big_img_name)
        big_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), big_img_name)
        
        small_img = self.load_image(small_img_path)
        small_mask = self.load_image(small_mask_path)
        
        big_img = self.load_image(big_img_path)
        big_mask = self.load_image(big_mask_path)
        
        small_img_pupil_xyr = torch.tensor(self.pupil_xyrs[small_img_name]) * self.res_mult
        small_img_iris_xyr = torch.tensor(self.iris_xyrs[small_img_name]) * self.res_mult
        
        big_img_pupil_xyr = torch.tensor(self.pupil_xyrs[big_img_name]) * self.res_mult
        big_img_iris_xyr = torch.tensor(self.iris_xyrs[big_img_name]) * self.res_mult
        
        xform_small_img = self.transform(small_img)
        xform_small_mask = self.transform(small_mask)
        xform_small_mask[xform_small_mask<0] = -1
        xform_small_mask[xform_small_mask>=0] = 1
        
        xform_big_img = self.transform(big_img)
        xform_big_mask = self.transform(big_mask)
        xform_big_mask[xform_big_mask<0] = -1
        xform_big_mask[xform_big_mask>=0] = 1
        
        return {'small_img': xform_small_img, 'small_mask' : xform_small_mask, 'big_img' : xform_big_img, 'big_mask' : xform_big_mask, 'small_img_pxyr': small_img_pupil_xyr, 'small_img_ixyr' : small_img_iris_xyr, 'big_img_pxyr' : big_img_pupil_xyr, 'big_img_ixyr' :  big_img_iris_xyr, 'identifier':self.pair_ids[index]}
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Setting Dataset for Min-max pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][max_bin_num][img_ind], self.bins[identifier][min_bin_num][img_ind]])
                self.pair_ids.append('_'.join(identifier.split('_')[:-1]))

class PairExtremeBinDataset(Dataset):
    def __init__(self, bins_path, parent_dir, max_pairs_inp = None, res_mult=1):
        super().__init__()
        
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.pupil_xyrs_path = os.path.join(parent_dir, 'WarsawPupilCircles.pkl')
        self.iris_xyrs_path = os.path.join(parent_dir, 'WarsawIrisCircles.pkl')
        self.max_pairs_inp = max_pairs_inp
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
        
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
        
        print('Setting Dataset for min-max and max-min pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][max_bin_num][img_ind], self.bins[identifier][min_bin_num][img_ind]])
                self.pair_ids.append('_'.join(identifier.split('_')[:-1]))
        
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][min_bin_num][img_ind], self.bins[identifier][max_bin_num][img_ind]])
                self.pair_ids.append('_'.join(identifier.split('_')[:-1]))
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        with open(self.pupil_xyrs_path, 'rb') as pupilFile:
            self.pupil_xyrs = pkl.load(pupilFile)
        
        with open(self.iris_xyrs_path, 'rb') as irisFile:
            self.iris_xyrs = pkl.load(irisFile)

    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        small_img_name = pair[0].strip()
        big_img_name = pair[1].strip()
        
        indiv_id1 = small_img_name.split('_')[0]
        indiv_id2 = big_img_name.split('_')[0]
        
        small_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), small_img_name)
        small_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id1), small_img_name)
        
        big_img_path = os.path.join(os.path.join(self.image_dir, indiv_id2), big_img_name)
        big_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), big_img_name)
        
        small_img = self.load_image(small_img_path)
        small_mask = self.load_image(small_mask_path)
        
        big_img = self.load_image(big_img_path)
        big_mask = self.load_image(big_mask_path)
        
        small_img_pupil_xyr = torch.tensor(self.pupil_xyrs[small_img_name]) * self.res_mult
        small_img_iris_xyr = torch.tensor(self.iris_xyrs[small_img_name]) * self.res_mult
        
        big_img_pupil_xyr = torch.tensor(self.pupil_xyrs[big_img_name]) * self.res_mult
        big_img_iris_xyr = torch.tensor(self.iris_xyrs[big_img_name]) * self.res_mult
        
        xform_small_img = self.transform(small_img)
        xform_small_mask = self.transform(small_mask)
        xform_small_mask[xform_small_mask<0] = -1
        xform_small_mask[xform_small_mask>=0] = 1
        
        xform_big_img = self.transform(big_img)
        xform_big_mask = self.transform(big_mask)
        xform_big_mask[xform_big_mask<0] = -1
        xform_big_mask[xform_big_mask>=0] = 1
        
        return {'small_img': xform_small_img, 'small_mask' : xform_small_mask, 'big_img' : xform_big_img, 'big_mask' : xform_big_mask, 'small_img_pxyr': small_img_pupil_xyr, 'small_img_ixyr' : small_img_iris_xyr, 'big_img_pxyr' : big_img_pupil_xyr, 'big_img_ixyr' :  big_img_iris_xyr, 'identifier':self.pair_ids[index] }
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Setting Dataset for Min-max pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][max_bin_num][img_ind], self.bins[identifier][min_bin_num][img_ind]])
                self.pair_ids.append('_'.join(identifier.split('_')[:-1]))
        
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num]))
            else:
                max_pairs = min(self.max_pairs_inp, min(len(self.bins[identifier][max_bin_num]), len(self.bins[identifier][min_bin_num])))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][min_bin_num][img_ind], self.bins[identifier][max_bin_num][img_ind]])
                self.pair_ids.append('_'.join(identifier.split('_')[:-1]))
        

class PairMinMaxBinDatasetPolar(Dataset):
    def __init__(self, bins_path, parent_dir, input_size=(240,320), polar_width=512, polar_height=64):
        super().__init__()
        
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.pupil_xyrs_path = os.path.join(parent_dir, 'WarsawPupilCircles.pkl')
        self.iris_xyrs_path = os.path.join(parent_dir, 'WarsawIrisCircles.pkl')
        self.max_pairs_inp = max_pairs_inp
        self.input_size = input_size
        self.polar_width = polar_width
        self.polar_height = polar_height
        
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
        
        print('Setting Dataset for Min-max pairs...')
        self.pairs = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            max_pairs = min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num]))
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][min_bin_num][img_ind], self.bins[identifier][max_bin_num][img_ind]])
         
        random.shuffle(self.pairs)  
        
        self.transform = transforms.Compose([
            transforms.Resize([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        with open(self.pupil_xyrs_path, 'rb') as pupilFile:
            self.pupil_xyrs = pkl.load(pupilFile)
        
        with open(self.iris_xyrs_path, 'rb') as irisFile:
            self.iris_xyrs = pkl.load(irisFile)
            
    def grid_sample(self, input, grid, interp_mode='bilinear'):

        # grid: [-1, 1]
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1)
        return torch.nn.functional.grid_sample(input, newgrid, mode=interp_mode)
        
    # Rubbersheet model-based Cartesian-to-polar transformation using bilinear interpolation from torch grid sample
    def cartToPol(self, image, mask, pupil_xyr, iris_xyr):
        with torch.no_grad():
            if pupil_xyr is None or iris_xyr is None:
                return None, None
            
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            width = image.shape[3]
            height = image.shape[2]

            polar_height = self.polar_height
            polar_width = self.polar_width

            pupil_xyr = torch.tensor(pupil_xyr).unsqueeze(0).float()
            iris_xyr = torch.tensor(iris_xyr).unsqueeze(0).float()
            
            theta = (2*pi*torch.linspace(1,polar_width,polar_width)/polar_width)
            pxCirclePoints = pupil_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width) #b x 512
            pyCirclePoints = pupil_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
            
            ixCirclePoints = iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)  #b x 512
            iyCirclePoints = iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512

            radius = (torch.linspace(0,polar_height,polar_height)/polar_height).reshape(-1, 1)  #64 x 1
            
            pxCoords = torch.matmul((1-radius), pxCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            pyCoords = torch.matmul((1-radius), pyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            
            ixCoords = torch.matmul(radius, ixCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            iyCoords = torch.matmul(radius, iyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512

            x = torch.clamp(pxCoords + ixCoords, 0, width-1).float()
            x_norm = (x/(width-1))*2 - 1 #b x 64 x 512

            y = torch.clamp(pyCoords + iyCoords, 0, height-1).float()
            y_norm = (y/(height-1))*2 - 1  #b x 64 x 512

            grid_sample_mat = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1)

            image_polar = self.grid_sample(image, grid_sample_mat, interp_mode='bilinear')
            mask_polar = self.grid_sample(mask, grid_sample_mat, interp_mode='nearest')

            return image_polar[0], mask_polar[0]
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        small_img_name = pair[0].strip()
        big_img_name = pair[1].strip()
        
        indiv_id1 = small_img_name.split('_')[0]
        indiv_id2 = big_img_name.split('_')[0]
        
        small_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), small_img_name)
        small_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id1), small_img_name)
        
        big_img_path = os.path.join(os.path.join(self.image_dir, indiv_id2), big_img_name)
        big_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), big_img_name)
        
        small_img = self.load_image(small_img_path)
        small_mask = self.load_image(small_mask_path)
        
        big_img = self.load_image(big_img_path)
        big_mask = self.load_image(big_mask_path)
        
        small_img_pupil_xyr = torch.tensor(self.pupil_xyrs[small_img_name])
        small_img_iris_xyr = torch.tensor(self.iris_xyrs[small_img_name])
        
        big_img_pupil_xyr = torch.tensor(self.pupil_xyrs[big_img_name])
        big_img_iris_xyr = torch.tensor(self.iris_xyrs[big_img_name])
        
        xform_small_img = self.transform(small_img)
        xform_small_mask = self.transform(small_mask)
        xform_small_mask[xform_small_mask<0] = -1
        xform_small_mask[xform_small_mask>=0] = 1
        
        xform_small_img_polar, xform_small_mask_polar = self.cartToPol(xform_small_img, xform_small_mask, small_img_pupil_xyr, small_img_iris_xyr)
        
        xform_big_img = self.transform(big_img)
        xform_big_mask = self.transform(big_mask)
        xform_big_mask[xform_big_mask<0] = -1
        xform_big_mask[xform_big_mask>=0] = 1
        
        xform_big_img_polar, xform_big_mask_polar = self.cartToPol(xform_big_img, xform_big_mask, big_img_pupil_xyr, big_img_iris_xyr)
        
        return {'small_img': xform_small_img_polar, 'small_mask' : xform_small_mask_polar, 'big_img' : xform_big_img_polar, 'big_mask' : xform_big_mask_polar, 'small_img_pxyr': small_img_pupil_xyr, 'small_img_ixyr' : small_img_iris_xyr, 'big_img_pxyr' : big_img_pupil_xyr, 'big_img_ixyr' :  big_img_iris_xyr }
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Setting Dataset for Min-max pairs...')
        self.pairs = []
        for identifier in tqdm(self.bins.keys()):
            min_bin_num = min(list(self.bins[identifier].keys()))
            max_bin_num = max(list(self.bins[identifier].keys()))
            random.shuffle(self.bins[identifier][min_bin_num])
            random.shuffle(self.bins[identifier][max_bin_num])
            if self.max_pairs_inp is None:
                max_pairs = min(len(self.bins[identifier][min_bin_num]), len(self.bins[identifier][max_bin_num]))
            else:
                max_pairs = self.max_pairs_inp
            for img_ind in range(max_pairs):
                self.pairs.append([self.bins[identifier][min_bin_num][img_ind], self.bins[identifier][max_bin_num][img_ind]])     

class PairFromBinDatasetBoth(Dataset):
    def __init__(self, bins_path, parent_dir, flip_data = True, res_mult=1):
        super().__init__()
        
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.pupil_xyrs_path = os.path.join(parent_dir, 'WarsawPupilCircles.pkl')
        self.iris_xyrs_path = os.path.join(parent_dir, 'WarsawIrisCircles.pkl')
        self.flip_data = flip_data   
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
        
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
        
        print('Initializing pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]])
                            self.pair_ids.append('_'.join(identifier.split('_')[:-1]))
        
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 > bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]])
                            self.pair_ids.append('_'.join(identifier.split('_')[:-1]))                
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        with open(self.pupil_xyrs_path, 'rb') as pupilFile:
            self.pupil_xyrs = pkl.load(pupilFile)
        
        with open(self.iris_xyrs_path, 'rb') as irisFile:
            self.iris_xyrs = pkl.load(irisFile)
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        small_img_name = pair[0].strip()
        big_img_name = pair[1].strip()
        
        indiv_id1 = small_img_name.split('_')[0]
        indiv_id2 = big_img_name.split('_')[0]
        
        small_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), small_img_name)
        small_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id1), small_img_name)
        
        big_img_path = os.path.join(os.path.join(self.image_dir, indiv_id2), big_img_name)
        big_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), big_img_name)
        
        small_img = self.load_image(small_img_path)
        small_mask = self.load_image(small_mask_path)
        
        big_img = self.load_image(big_img_path)
        big_mask = self.load_image(big_mask_path)
        
        small_img_pupil_xyr = torch.tensor(self.pupil_xyrs[small_img_name]) * self.res_mult
        small_img_iris_xyr = torch.tensor(self.iris_xyrs[small_img_name]) * self.res_mult
        
        big_img_pupil_xyr = torch.tensor(self.pupil_xyrs[big_img_name]) * self.res_mult
        big_img_iris_xyr = torch.tensor(self.iris_xyrs[big_img_name]) * self.res_mult
        
        if self.flip_data and random.random() < 0.5:
            small_img = small_img.transpose(Image.FLIP_LEFT_RIGHT) 
            small_mask = small_mask.transpose(Image.FLIP_LEFT_RIGHT)
            big_img = big_img.transpose(Image.FLIP_LEFT_RIGHT) 
            big_mask = big_mask.transpose(Image.FLIP_LEFT_RIGHT)
            small_img_pupil_xyr[0] = self.input_size[1] - small_img_pupil_xyr[0]
            big_img_pupil_xyr[0] = self.input_size[1] - big_img_pupil_xyr[0]
            small_img_iris_xyr[0] = self.input_size[1] - small_img_iris_xyr[0]
            big_img_iris_xyr[0] = self.input_size[1] - big_img_iris_xyr[0]
        
        xform_small_img = self.transform(small_img)
        xform_small_mask = self.transform(small_mask)
        xform_small_mask[xform_small_mask<0] = -1
        xform_small_mask[xform_small_mask>=0] = 1
        
        xform_big_img = self.transform(big_img)
        xform_big_mask = self.transform(big_mask)
        xform_big_mask[xform_big_mask<0] = -1
        xform_big_mask[xform_big_mask>=0] = 1
        
        
        return {'small_img': xform_small_img, 'small_mask' : xform_small_mask, 'big_img' : xform_big_img, 'big_mask' : xform_big_mask, 'small_img_pxyr': small_img_pupil_xyr, 'small_img_ixyr' : small_img_iris_xyr, 'big_img_pxyr' : big_img_pupil_xyr, 'big_img_ixyr' :  big_img_iris_xyr, 'identifier':self.pair_ids[index]}
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Resetting Pairs....')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]])
                            self.pair_ids.append('_'.join(identifier.split('_')[:-1]))
        
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 > bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]])
                            self.pair_ids.append('_'.join(identifier.split('_')[:-1]))    
                            
class PairFromBinDatasetSB(Dataset):
    def __init__(self, bins_path, parent_dir, flip_data = True, res_mult=1):
        super().__init__()
        
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.pupil_xyrs_path = os.path.join(parent_dir, 'WarsawPupilCircles.pkl')
        self.iris_xyrs_path = os.path.join(parent_dir, 'WarsawIrisCircles.pkl')
        self.flip_data = flip_data   
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
        
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
        
        print('Initializing pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]])  
                            self.pair_ids.append('_'.join(identifier.split('_')[:-1]))          
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        with open(self.pupil_xyrs_path, 'rb') as pupilFile:
            self.pupil_xyrs = pkl.load(pupilFile)
        
        with open(self.iris_xyrs_path, 'rb') as irisFile:
            self.iris_xyrs = pkl.load(irisFile)
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        small_img_name = pair[0].strip()
        big_img_name = pair[1].strip()
        
        indiv_id1 = small_img_name.split('_')[0]
        indiv_id2 = big_img_name.split('_')[0]
        
        small_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), small_img_name)
        small_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id1), small_img_name)
        
        big_img_path = os.path.join(os.path.join(self.image_dir, indiv_id2), big_img_name)
        big_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), big_img_name)
        
        small_img = self.load_image(small_img_path)
        small_mask = self.load_image(small_mask_path)
        
        big_img = self.load_image(big_img_path)
        big_mask = self.load_image(big_mask_path)
        
        small_img_pupil_xyr = torch.tensor(self.pupil_xyrs[small_img_name]) * self.res_mult
        small_img_iris_xyr = torch.tensor(self.iris_xyrs[small_img_name]) * self.res_mult
        
        big_img_pupil_xyr = torch.tensor(self.pupil_xyrs[big_img_name]) * self.res_mult
        big_img_iris_xyr = torch.tensor(self.iris_xyrs[big_img_name]) * self.res_mult
        
        if self.flip_data and random.random() < 0.5:
            small_img = small_img.transpose(Image.FLIP_LEFT_RIGHT) 
            small_mask = small_mask.transpose(Image.FLIP_LEFT_RIGHT)
            big_img = big_img.transpose(Image.FLIP_LEFT_RIGHT) 
            big_mask = big_mask.transpose(Image.FLIP_LEFT_RIGHT)
            small_img_pupil_xyr[0] = self.input_size[1] - small_img_pupil_xyr[0]
            big_img_pupil_xyr[0] = self.input_size[1] - big_img_pupil_xyr[0]
            small_img_iris_xyr[0] = self.input_size[1] - small_img_iris_xyr[0]
            big_img_iris_xyr[0] = self.input_size[1] - big_img_iris_xyr[0]
        
        xform_small_img = self.transform(small_img)
        xform_small_mask = self.transform(small_mask)
        xform_small_mask[xform_small_mask<0] = -1
        xform_small_mask[xform_small_mask>=0] = 1
        
        xform_big_img = self.transform(big_img)
        xform_big_mask = self.transform(big_mask)
        xform_big_mask[xform_big_mask<0] = -1
        xform_big_mask[xform_big_mask>=0] = 1
        
        return {'small_img': xform_small_img, 'small_mask' : xform_small_mask, 'big_img' : xform_big_img, 'big_mask' : xform_big_mask, 'small_img_pxyr': small_img_pupil_xyr, 'small_img_ixyr' : small_img_iris_xyr, 'big_img_pxyr' : big_img_pupil_xyr, 'big_img_ixyr' :  big_img_iris_xyr, 'identifier':self.pair_ids[index] }
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Resetting Pairs....')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]])  
                            self.pair_ids.append('_'.join(identifier.split('_')[:-1]))          
                            
class PairFromBinDatasetBS(Dataset):
    def __init__(self, bins_path, parent_dir, flip_data = True, res_mult=1):
        super().__init__()
        
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.pupil_xyrs_path = os.path.join(parent_dir, 'WarsawPupilCircles.pkl')
        self.iris_xyrs_path = os.path.join(parent_dir, 'WarsawIrisCircles.pkl')
        self.flip_data = flip_data   
        self.input_size=(int(240*res_mult),int(320*res_mult))
        self.res_mult = res_mult
        
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
        
        print('Initializing pairs...')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 > bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]]) 
                            self.pair_ids.append('_'.join(identifier.split('_')[:-1]))           
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        with open(self.pupil_xyrs_path, 'rb') as pupilFile:
            self.pupil_xyrs = pkl.load(pupilFile)
        
        with open(self.iris_xyrs_path, 'rb') as irisFile:
            self.iris_xyrs = pkl.load(irisFile)
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        small_img_name = pair[0].strip()
        big_img_name = pair[1].strip()
        
        indiv_id1 = small_img_name.split('_')[0]
        indiv_id2 = big_img_name.split('_')[0]
        
        small_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), small_img_name)
        small_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id1), small_img_name)
        
        big_img_path = os.path.join(os.path.join(self.image_dir, indiv_id2), big_img_name)
        big_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), big_img_name)
        
        small_img = self.load_image(small_img_path)
        small_mask = self.load_image(small_mask_path)
        
        big_img = self.load_image(big_img_path)
        big_mask = self.load_image(big_mask_path)
        
        small_img_pupil_xyr = torch.tensor(self.pupil_xyrs[small_img_name]) * self.res_mult
        small_img_iris_xyr = torch.tensor(self.iris_xyrs[small_img_name]) * self.res_mult
        
        big_img_pupil_xyr = torch.tensor(self.pupil_xyrs[big_img_name]) * self.res_mult
        big_img_iris_xyr = torch.tensor(self.iris_xyrs[big_img_name]) * self.res_mult
        
        if self.flip_data and random.random() < 0.5:
            small_img = small_img.transpose(Image.FLIP_LEFT_RIGHT) 
            small_mask = small_mask.transpose(Image.FLIP_LEFT_RIGHT)
            big_img = big_img.transpose(Image.FLIP_LEFT_RIGHT) 
            big_mask = big_mask.transpose(Image.FLIP_LEFT_RIGHT)
            small_img_pupil_xyr[0] = self.input_size[1] - small_img_pupil_xyr[0]
            big_img_pupil_xyr[0] = self.input_size[1] - big_img_pupil_xyr[0]
            small_img_iris_xyr[0] = self.input_size[1] - small_img_iris_xyr[0]
            big_img_iris_xyr[0] = self.input_size[1] - big_img_iris_xyr[0]
        
        xform_small_img = self.transform(small_img)
        xform_small_mask = self.transform(small_mask)
        xform_small_mask[xform_small_mask<0] = -1
        xform_small_mask[xform_small_mask>=0] = 1
        
        xform_big_img = self.transform(big_img)
        xform_big_mask = self.transform(big_mask)
        xform_big_mask[xform_big_mask<0] = -1
        xform_big_mask[xform_big_mask>=0] = 1
        
        
        return {'small_img': xform_small_img, 'small_mask' : xform_small_mask, 'big_img' : xform_big_img, 'big_mask' : xform_big_mask, 'small_img_pxyr': small_img_pupil_xyr, 'small_img_ixyr' : small_img_iris_xyr, 'big_img_pxyr' : big_img_pupil_xyr, 'big_img_ixyr' :  big_img_iris_xyr, 'identifier': self.pair_ids[index] }
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Resetting Pairs....')
        self.pairs = []
        self.pair_ids = []
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 > bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]]) 
                            self.pair_ids.append('_'.join(identifier.split('_')[:-1])) 

class PairFromBinDatasetPolar(Dataset):
    def __init__(self, bins_path, parent_dir, flip_data = True, input_size=(240,320), polar_width=512, polar_height=64):
        super().__init__()
        
        self.image_dir = os.path.join(parent_dir, 'WarsawData')
        self.mask_dir = os.path.join(parent_dir, 'WarsawMask')
        self.pupil_xyrs_path = os.path.join(parent_dir, 'WarsawPupilCircles.pkl')
        self.iris_xyrs_path = os.path.join(parent_dir, 'WarsawIrisCircles.pkl')
        self.flip_data = flip_data   
        self.input_size = input_size
        self.polar_width = polar_width
        self.polar_height = polar_height
        
        with open(bins_path, 'rb') as binsFile:
            self.bins = pkl.load(binsFile)
        
        print('Initializing pairs...')
        self.pairs = []
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]])            
        
        self.transform = transforms.Compose([
            transforms.Resize([self.input_size[0], self.input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        
        with open(self.pupil_xyrs_path, 'rb') as pupilFile:
            self.pupil_xyrs = pkl.load(pupilFile)
        
        with open(self.iris_xyrs_path, 'rb') as irisFile:
            self.iris_xyrs = pkl.load(irisFile)
            
    def grid_sample(self, input, grid, interp_mode='bilinear'):

        # grid: [-1, 1]
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1)
        return torch.nn.functional.grid_sample(input, newgrid, mode=interp_mode)
        
    # Rubbersheet model-based Cartesian-to-polar transformation using bilinear interpolation from torch grid sample
    def cartToPol(self, image, mask, pupil_xyr, iris_xyr):
        with torch.no_grad():
            if pupil_xyr is None or iris_xyr is None:
                return None, None
            
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            width = image.shape[3]
            height = image.shape[2]

            polar_height = self.polar_height
            polar_width = self.polar_width

            pupil_xyr = torch.tensor(pupil_xyr).unsqueeze(0).float()
            iris_xyr = torch.tensor(iris_xyr).unsqueeze(0).float()
            
            theta = (2*pi*torch.linspace(1,polar_width,polar_width)/polar_width)
            pxCirclePoints = pupil_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width) #b x 512
            pyCirclePoints = pupil_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512
            
            ixCirclePoints = iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, polar_width)  #b x 512
            iyCirclePoints = iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, polar_width)  #b x 512

            radius = (torch.linspace(0,polar_height,polar_height)/polar_height).reshape(-1, 1)  #64 x 1
            
            pxCoords = torch.matmul((1-radius), pxCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            pyCoords = torch.matmul((1-radius), pyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            
            ixCoords = torch.matmul(radius, ixCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512
            iyCoords = torch.matmul(radius, iyCirclePoints.reshape(-1, 1, polar_width)) # b x 64 x 512

            x = torch.clamp(pxCoords + ixCoords, 0, width-1).float()
            x_norm = (x/(width-1))*2 - 1 #b x 64 x 512

            y = torch.clamp(pyCoords + iyCoords, 0, height-1).float()
            y_norm = (y/(height-1))*2 - 1  #b x 64 x 512

            grid_sample_mat = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1)

            image_polar = self.grid_sample(image, grid_sample_mat, interp_mode='bilinear')
            mask_polar = self.grid_sample(mask, grid_sample_mat, interp_mode='nearest')

            return image_polar[0], mask_polar[0]
    
    def load_image(self, file):
        return Image.open(file).convert('L')
        
    def __getitem__(self, index):
    
        pair = self.pairs[index]
        
        small_img_name = pair[0].strip()
        big_img_name = pair[1].strip()
        
        indiv_id1 = small_img_name.split('_')[0]
        indiv_id2 = big_img_name.split('_')[0]
        
        small_img_path = os.path.join(os.path.join(self.image_dir, indiv_id1), small_img_name)
        small_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id1), small_img_name)
        
        big_img_path = os.path.join(os.path.join(self.image_dir, indiv_id2), big_img_name)
        big_mask_path = os.path.join(os.path.join(self.mask_dir, indiv_id2), big_img_name)
        
        small_img = self.load_image(small_img_path)
        small_mask = self.load_image(small_mask_path)
        
        big_img = self.load_image(big_img_path)
        big_mask = self.load_image(big_mask_path)
        
        small_img_pupil_xyr = torch.tensor(self.pupil_xyrs[small_img_name])
        small_img_iris_xyr = torch.tensor(self.iris_xyrs[small_img_name])
        
        big_img_pupil_xyr = torch.tensor(self.pupil_xyrs[big_img_name])
        big_img_iris_xyr = torch.tensor(self.iris_xyrs[big_img_name])
        
        if self.flip_data and random.random() < 0.5:
            small_img = small_img.transpose(Image.FLIP_LEFT_RIGHT) 
            small_mask = small_mask.transpose(Image.FLIP_LEFT_RIGHT)
            big_img = big_img.transpose(Image.FLIP_LEFT_RIGHT) 
            big_mask = big_mask.transpose(Image.FLIP_LEFT_RIGHT)
            small_img_pupil_xyr[0] = self.input_size[1] - small_img_pupil_xyr[0]
            big_img_pupil_xyr[0] = self.input_size[1] - big_img_pupil_xyr[0]
            small_img_iris_xyr[0] = self.input_size[1] - small_img_iris_xyr[0]
            big_img_iris_xyr[0] = self.input_size[1] - big_img_iris_xyr[0]
        
        xform_small_img = self.transform(small_img)
        xform_small_mask = self.transform(small_mask)
        xform_small_mask[xform_small_mask<0] = -1
        xform_small_mask[xform_small_mask>=0] = 1
        
        xform_small_img_polar, xform_small_mask_polar = self.cartToPol(xform_small_img, xform_small_mask, small_img_pupil_xyr, small_img_iris_xyr)
        
        xform_big_img = self.transform(big_img)
        xform_big_mask = self.transform(big_mask)
        xform_big_mask[xform_big_mask<0] = -1
        xform_big_mask[xform_big_mask>=0] = 1
        
        xform_big_img_polar, xform_big_mask_polar = self.cartToPol(xform_big_img, xform_big_mask, big_img_pupil_xyr, big_img_iris_xyr)        
        
        return {'small_img': xform_small_img_polar, 'small_mask' : xform_small_mask_polar, 'big_img' : xform_big_img_polar, 'big_mask' : xform_big_mask_polar, 'small_img_pxyr': small_img_pupil_xyr, 'small_img_ixyr' : small_img_iris_xyr, 'big_img_pxyr' : big_img_pupil_xyr, 'big_img_ixyr' :  big_img_iris_xyr }
    
    def __len__(self):
        return len(self.pairs)
    
    def reset(self):
        print('Resetting Pairs....')
        self.pairs = []
        for identifier in tqdm(self.bins.keys()):
            for bin_num_1 in self.bins[identifier].keys():
                for bin_num_2 in self.bins[identifier].keys():
                    if bin_num_1 < bin_num_2:
                        random.shuffle(self.bins[identifier][bin_num_1])
                        random.shuffle(self.bins[identifier][bin_num_2])
                        max_pairs = min(len(self.bins[identifier][bin_num_1]), len(self.bins[identifier][bin_num_2]))
                        for img_ind in range(max_pairs):
                            self.pairs.append([self.bins[identifier][bin_num_1][img_ind], self.bins[identifier][bin_num_2][img_ind]])      