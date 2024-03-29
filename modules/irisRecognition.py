import numpy as np
import cv2
import scipy.io, scipy.signal
import argparse
from modules.network import UNet
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from PIL import Image
from skimage import img_as_bool
import math
from math import pi
from modules.network import UNet_radius_center_denseconv
from scipy.integrate import solve_bvp
import os

class irisRecognition(object):
    def __init__(self, cfg, use_hough=True):

        # cParsing the config file
        self.use_hough = use_hough
        self.polar_height = cfg["polar_height"]
        self.polar_width = cfg["polar_width"]
        self.angles = angles = np.arange(0, 2 * np.pi, 2 * np.pi / self.polar_width)
        self.cos_angles = np.zeros((self.polar_width))
        self.sin_angles = np.zeros((self.polar_width))
        for i in range(self.polar_width):
            self.cos_angles[i] = np.cos(self.angles[i])
            self.sin_angles[i] = np.sin(self.angles[i])
        self.filter_size = cfg["recog_filter_size"]
        self.num_filters = cfg["recog_num_filters"]
        self.max_shift = cfg["recog_max_shift"]
        self.cuda = cfg["cuda"]
        self.mod_ccnet_model_path = cfg["modified_ccnet_model_path"]
        self.ccnet_model_path = cfg["ccnet_model_path"]
        self.filter = scipy.io.loadmat(cfg["recog_bsif_dir"]+'ICAtextureFilters_{0}x{1}_{2}bit.mat'.format(self.filter_size, self.filter_size, self.num_filters))['ICAtextureFilters']
        if self.use_hough == True:
            self.iris_hough_param1 = cfg["iris_hough_param1"]
            self.iris_hough_param2 = cfg["iris_hough_param2"]
            self.iris_hough_margin = cfg["iris_hough_margin"]
            self.pupil_hough_param1 = cfg["pupil_hough_param1"]
            self.pupil_hough_param2 = cfg["pupil_hough_param2"]
            self.pupil_hough_minimum = cfg["pupil_hough_minimum"]
            self.pupil_iris_max_ratio = cfg["pupil_iris_max_ratio"]
            self.max_pupil_iris_shift = cfg["max_pupil_iris_shift"]
        self.visMinAgreedBits = cfg["vis_min_agreed_bits"]
        self.vis_mode = cfg["vis_mode"]

        # Loading the CCNet
        self.CCNET_INPUT_SIZE = (320,240)
        self.CCNET_NUM_CHANNELS = 1
        self.CCNET_NUM_CLASSES = 2
        self.model = UNet(self.CCNET_NUM_CLASSES, self.CCNET_NUM_CHANNELS)
        self.mod_model = UNet_radius_center_denseconv(self.CCNET_NUM_CLASSES, self.CCNET_NUM_CHANNELS, num_params=6, width=8, n_convs=10, is_bn = False, dense_bn = False)
        if self.cuda:
            self.device = torch.device('cuda')
            #torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.model = self.model.cuda()
            self.mod_model = self.mod_model.cuda()
        else:
            self.device = torch.device('cpu')
            #torch.set_default_tensor_type('torch.FloatTensor')
        if self.ccnet_model_path:
            try:
                if self.cuda:
                    self.model.load_state_dict(torch.load(self.ccnet_model_path, map_location=torch.device('cuda')))
                else:
                    self.model.load_state_dict(torch.load(self.ccnet_model_path, map_location=torch.device('cpu')))
                    # print("model state loaded")
            except AssertionError:
                print("assertion error")
                self.model.load_state_dict(torch.load(self.ccnet_model_path,
                    map_location = lambda storage, loc: storage))
        if not self.use_hough:
            if self.mod_ccnet_model_path:
                try:
                    if self.cuda:
                        self.mod_model.load_state_dict(torch.load(self.mod_ccnet_model_path, map_location=torch.device('cuda')))
                    else:
                        self.mod_model.load_state_dict(torch.load(self.mod_ccnet_model_path, map_location=torch.device('cpu')))
                        # print("model state loaded")
                except AssertionError:
                    print("assertion error")
                    self.mod_model.load_state_dict(torch.load(self.mod_ccnet_model_path,
                        map_location = lambda storage, loc: storage))
                
        self.model.eval()
        self.mod_model.eval()
        self.softmax = nn.LogSoftmax(dim=1)
        self.input_transform = Compose([ToTensor(),])
        # print("irisRecognition class: initialized")

        # Misc
        self.se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        self.sk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        self.ISO_RES = (640,480)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
        
    
    ### Use this function for a faster estimation of mask and circle parameters. When use_hough is False, this function uses both the circle approximation and the mask from MCCNet
    def segment_and_circApprox(self, image):
        if self.use_hough:
            pred = self.segment(image)
            pupil_xyr, iris_xyr = self.circApprox(pred)
            return pred, pupil_xyr, iris_xyr
        else:
            w,h = image.size
            image = cv2.resize(np.array(image), self.CCNET_INPUT_SIZE, cv2.INTER_LINEAR)
            w_mult = w/self.CCNET_INPUT_SIZE[0]
            h_mult = h/self.CCNET_INPUT_SIZE[1]

            outputs, inp_xyr_t = self.mod_model(Variable(self.input_transform(image).unsqueeze(0).to(self.device)))

            #Circle params
            inp_xyr = inp_xyr_t.tolist()[0]
            pupil_x = int(inp_xyr[0] * w_mult)
            pupil_y = int(inp_xyr[1] * h_mult)
            pupil_r = int(inp_xyr[2] * max(w_mult, h_mult))
            iris_x = int(inp_xyr[3] * w_mult)
            iris_y = int(inp_xyr[4] * h_mult)
            iris_r = int(inp_xyr[5] * max(w_mult, h_mult))

            #Mask
            logprob = self.softmax(outputs).data.cpu().numpy()
            pred = np.argmax(logprob, axis=1)*255
            pred = Image.fromarray(pred[0].astype(np.uint8))
            pred = np.array(pred)
        
            # Optional: uncomment the following lines to take only the biggest blob returned by CCNet
            '''
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(pred, connectivity=4)
            if nb_components > 1:
                sizes = stats[:, -1] 
                max_label = 1
                max_size = sizes[1]    
                for i in range(2, nb_components):
                    if sizes[i] > max_size:
                        max_label = i
                        max_size = sizes[i]

                pred = np.zeros(output.shape)
                pred[output == max_label] = 255
                pred = np.asarray(pred, dtype=np.uint8)
            '''

            # Resize the mask to the original image size
            pred = img_as_bool(cv2.resize(np.array(pred), (w,h), cv2.INTER_NEAREST))

            return pred, np.array([pupil_x,pupil_y,pupil_r]), np.array([iris_x,iris_y,iris_r])

    def segment(self,image):

        w,h = image.size
        image = cv2.resize(np.array(image), self.CCNET_INPUT_SIZE, cv2.INTER_LINEAR)

        outputs = self.model(Variable(self.input_transform(image).unsqueeze(0).to(self.device)))
        logprob = self.softmax(outputs).data.cpu().numpy()
        pred = np.argmax(logprob, axis=1)*255
        pred = Image.fromarray(pred[0].astype(np.uint8))
        pred = np.array(pred)

        # Optional: uncomment the following lines to take only the biggest blob returned by CCNet
        '''
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(pred, connectivity=4)
        if nb_components > 1:
            sizes = stats[:, -1] 
            max_label = 1
            max_size = sizes[1]    
            for i in range(2, nb_components):
                if sizes[i] > max_size:
                    max_label = i
                    max_size = sizes[i]

            pred = np.zeros(output.shape)
            pred[output == max_label] = 255
            pred = np.asarray(pred, dtype=np.uint8)
        '''

        # Resize the mask to the original image size
        pred = img_as_bool(cv2.resize(np.array(pred), (w,h), cv2.INTER_NEAREST))

        return pred


    def segmentVis(self,im,mask,pupil_xyr,iris_xyr):

        imVis = np.stack((np.array(im),)*3, axis=-1)
        imVis[:,:,1] = np.clip(imVis[:,:,1] + 96*mask,0,255)
        imVis = cv2.circle(imVis, (pupil_xyr[0],pupil_xyr[1]), pupil_xyr[2], (0, 0, 255), 2)
        imVis = cv2.circle(imVis, (iris_xyr[0],iris_xyr[1]), iris_xyr[2], (255, 0, 0), 2)

        return imVis
    

    def circApprox(self,mask=None,image=None):
        if self.use_hough and mask is None:
            print('Please provide mask if you want to use hough transform')
        if (not self.use_hough) and image is None:
            print('Please provide image if you want to use the mccnet model') 
        if self.use_hough and (mask is not None):
            # Iris boundary approximation
            mask_for_iris = 255*(1 - np.uint8(mask))
            iris_indices = np.where(mask_for_iris == 0)
            if len(iris_indices[0]) == 0:
                return None, None
            y_span = max(iris_indices[0]) - min(iris_indices[0])
            x_span = max(iris_indices[1]) - min(iris_indices[1])

            iris_radius_estimate = np.max((x_span,y_span)) // 2
            iris_circle = cv2.HoughCircles(mask_for_iris, cv2.HOUGH_GRADIENT, 1, 50,
                                        param1=self.iris_hough_param1,
                                        param2=self.iris_hough_param2,
                                        minRadius=iris_radius_estimate-self.iris_hough_margin,
                                        maxRadius=iris_radius_estimate+self.iris_hough_margin)
            if iris_circle is None:
                return None, None
            iris_x, iris_y, iris_r = np.rint(np.array(iris_circle[0][0])).astype(int)
            
            
            # Pupil boundary approximation
            pupil_circle = cv2.HoughCircles(mask_for_iris, cv2.HOUGH_GRADIENT, 1, 50,
                                            param1=self.pupil_hough_param1,
                                            param2=self.pupil_hough_param2,
                                            minRadius=self.pupil_hough_minimum,
                                            maxRadius=np.int(self.pupil_iris_max_ratio*iris_r))
            if pupil_circle is None:
                return None, None
            pupil_x, pupil_y, pupil_r = np.rint(np.array(pupil_circle[0][0])).astype(int)
            
            if np.sqrt((pupil_x-iris_x)**2+(pupil_y-iris_y)**2) > self.max_pupil_iris_shift:
                pupil_x = iris_x
                pupil_y = iris_y
                pupil_r = iris_r // 3

            return np.array([pupil_x,pupil_y,pupil_r]), np.array([iris_x,iris_y,iris_r])
        elif (not self.use_hough) and (image is not None):
            w,h = image.size
            image = cv2.resize(np.array(image), self.CCNET_INPUT_SIZE, cv2.INTER_CUBIC)
            w_mult = w/self.CCNET_INPUT_SIZE[0]
            h_mult = h/self.CCNET_INPUT_SIZE[1]

            outputs, inp_xyr_t = self.mod_model(Variable(self.input_transform(image).unsqueeze(0).to(self.device)))

            #Circle params
            inp_xyr = inp_xyr_t.tolist()[0]
            pupil_x = round(inp_xyr[0] * w_mult)
            pupil_y = round(inp_xyr[1] * h_mult)
            pupil_r = round(inp_xyr[2] * max(w_mult, h_mult))
            iris_x = round(inp_xyr[3] * w_mult)
            iris_y = round(inp_xyr[4] * h_mult)
            iris_r = round(inp_xyr[5] * max(w_mult, h_mult))

            return np.array([pupil_x,pupil_y,pupil_r]).astype(int), np.array([iris_x,iris_y,iris_r]).astype(int)

    
    def grid_sample(self, input, grid, interp_mode='bilinear'):

        # grid: [-1, 1]
        N, C, H, W = input.shape
        gridx = grid[:, :, :, 0]
        gridy = grid[:, :, :, 1]
        gridx = ((gridx + 1) / 2 *  W - 0.5) / (W - 1) * 2 - 1
        gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gridx, gridy], dim=-1)
        return torch.nn.functional.grid_sample(input, newgrid, mode=interp_mode, align_corners=True)
        
    # Rubbersheet model-based Cartesian-to-polar transformation using bilinear interpolation from torch grid sample
    
    def cartToPol(self, image, mask, pupil_xyr, iris_xyr):
        with torch.no_grad():
            if pupil_xyr is None or iris_xyr is None:
                return None, None
            
            image = ToTensor()(image).unsqueeze(0) * 255
            mask = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0)
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

            radius = (torch.linspace(0,self.polar_height,self.polar_height)/self.polar_height).reshape(-1, 1)  #64 x 1
            
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
            image_polar = torch.clamp(torch.round(image_polar), min=0, max=255)
            mask_polar = self.grid_sample(mask, grid_sample_mat, interp_mode='nearest')
            mask_polar = (mask_polar>0.5).long() * 255

            return (image_polar[0][0].cpu().numpy()).astype(np.uint8), mask_polar[0][0].cpu().numpy().astype(np.uint8)
    '''
    def bioMechCartToPol(self, image, mask, pupil_xyr, iris_xyr, radiusBiomech):
         with torch.no_grad():
            if pupil_xyr is None or iris_xyr is None:
                return None, None
            
            image = ToTensor()(image).unsqueeze(0) * 255
            mask = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0)
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

            
            radius = torch.tensor(radiusBiomech).reshape(-1, 1).float()  #64 x 1
            #print(radius)
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
            image_polar = torch.clamp(torch.round(image_polar), min=0, max=255)
            mask_polar = self.grid_sample(mask, grid_sample_mat, interp_mode='nearest')
            mask_polar = (mask_polar>0.5).long() * 255

            return (image_polar[0][0].cpu().numpy()).astype(np.uint8), mask_polar[0][0].cpu().numpy().astype(np.uint8)
    '''
    def bioMechCartToPol(self, image, mask, pupil_xyr, iris_xyr, radiusBiomech):
        '''   
        if pupil_xyr is None or iris_xyr is None:
            return None, None
       
        image = np.array(image)
        height, width = image.shape
        mask = np.array(mask)

        image_polar = np.zeros((self.polar_height, self.polar_width), np.uint8)
        mask_polar = np.zeros((self.polar_height, self.polar_width), np.uint8)

        theta = 2*pi*np.linspace(1,self.polar_width,self.polar_width)/self.polar_width
        pxCirclePoints = pupil_xyr[0] + pupil_xyr[2]*np.cos(theta)
        pyCirclePoints = pupil_xyr[1] + pupil_xyr[2]*np.sin(theta)
        
        ixCirclePoints = iris_xyr[0] + iris_xyr[2]*np.cos(theta)
        iyCirclePoints = iris_xyr[1] + iris_xyr[2]*np.sin(theta)

        radius = np.float32()
        for j in range(self.polar_width):
            x = (np.clip(0,width-1,np.around((1-radius) * pxCirclePoints[j] + radius * ixCirclePoints[j]))).astype(int)
            y = (np.clip(0,height-1,np.around((1-radius) * pyCirclePoints[j] + radius * iyCirclePoints[j]))).astype(int)
            
            for i in range(self.polar_height):
                if (x[i] > 0 and x[i] < width and y[i] > 0 and y[i] < height): 
                    image_polar[i][j] = image[y[i]][x[i]]
                    mask_polar[i][j] = 255*mask[y[i]][x[i]]
        return image_polar, mask_polar
        '''         
        with torch.no_grad():
            if pupil_xyr is None or iris_xyr is None:
                return None, None
            
            image = ToTensor()(image).unsqueeze(0) * 255
            mask = torch.tensor(mask).float().unsqueeze(0).unsqueeze(0)
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

            radius = torch.tensor(radiusBiomech).float().reshape(-1, 1)  #64 x 1
            
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
            image_polar = torch.clamp(torch.round(image_polar), min=0, max=255)
            mask_polar = self.grid_sample(mask, grid_sample_mat, interp_mode='nearest')
            mask_polar = (mask_polar>0.5).long() * 255

            return (image_polar[0][0].cpu().numpy()).astype(np.uint8), mask_polar[0][0].cpu().numpy().astype(np.uint8)
        
    def findRadiusBiomech(self, rpo0, rpo1, rso0, rso1):
        rp0 = (rpo0/rso0) * 6
        rp1 = (rpo1/rso1) * 6
        rs = np.linspace(rp0, 6, self.polar_height)
        dilation = rp1 - rp0
        def bc(ya, yb):
            return np.array([ya[0]-dilation, yb[0]])

        def f(x, y):
            gamma = (2.97/4)
            v = 0.49
            st_up = -y[1]/x  + (gamma * y[0])/(x ** 2) + ((1-v*gamma)/(2*x)) * ((y[1])**2) + (((v-1)*gamma)/(2*x)) * ((y[0]/x) ** 2) - ((v * gamma * (y[0] ** 2))/(x**3)) + ((v * gamma * y[0] * y[1])/(x ** 2))
            st = st_up/(1-y[1])
            return np.vstack((y[1], st))

        ui = np.zeros((2, rs.shape[0]))
        res = solve_bvp(f, bc, rs, ui)
        us = res.sol(rs)[0]
        rs_biomech = rs + us
        rs_biomech = rs_biomech - rs_biomech.min()
        rs_biomech = rs_biomech / rs_biomech.max()
        return rs_biomech
    
    def plotBioMechPoints(self, image_name, image, mask, pupil_xyr, iris_xyr, radiusBiomech):
        
        if pupil_xyr is None:
            return None, None
       
        image = np.array(image)
        image2 = np.copy(image)

        height, width = image.shape
        mask = np.array(mask)

        image_polar = np.zeros((self.polar_height, self.polar_width), np.uint8)
        mask_polar = np.zeros((self.polar_height, self.polar_width), np.uint8)

        theta = 2*pi*np.linspace(1,self.polar_width,self.polar_width)/self.polar_width
        pxCirclePoints = pupil_xyr[0] + pupil_xyr[2]*np.cos(theta)
        pyCirclePoints = pupil_xyr[1] + pupil_xyr[2]*np.sin(theta)
        
        ixCirclePoints = iris_xyr[0] + iris_xyr[2]*np.cos(theta)
        iyCirclePoints = iris_xyr[1] + iris_xyr[2]*np.sin(theta)

        radius = np.linspace(0,self.polar_height,self.polar_height)/self.polar_height

        for j in range(self.polar_width):
            x = (np.clip(0,width-1,np.around((1-radius) * pxCirclePoints[j] + radius * ixCirclePoints[j]))).astype(int)
            y = (np.clip(0,height-1,np.around((1-radius) * pyCirclePoints[j] + radius * iyCirclePoints[j]))).astype(int)
            
            x_bio = (np.clip(0,width-1,np.around((1-radiusBiomech) * pxCirclePoints[j] + radiusBiomech * ixCirclePoints[j]))).astype(int)
            y_bio = (np.clip(0,height-1,np.around((1-radiusBiomech) * pyCirclePoints[j] + radiusBiomech * iyCirclePoints[j]))).astype(int)

            for i in range(self.polar_height):
                if i % 5 == 0:
                    if (x[i] > 0 and x[i] < width and y[i] > 0 and y[i] < height): 
                        image[y[i]][x[i]] = 0
                        image2[y_bio[i]][x_bio[i]] = 0
        
        if not os.path.exists('./biomech_samples/'):
            os.mkdir('./biomech_samples/')
        
        cv2.imwrite('./biomech_samples/'+image_name, image)
        cv2.imwrite('./biomech_samples/bio_'+image_name, image2)

        return True
    
    # Previous implementation that uses nearest neighbor interpolation
    # Rubbersheet model-based Cartesian-to-polar transformation
    def cartToPol_prev(self, image, mask, pupil_xyr, iris_xyr):
        
        if pupil_xyr is None:
            return None, None
       
        image = np.array(image)
        height, width = image.shape
        mask = np.array(mask)

        image_polar = np.zeros((self.polar_height, self.polar_width), np.uint8)
        mask_polar = np.zeros((self.polar_height, self.polar_width), np.uint8)

        theta = 2*pi*np.linspace(1,self.polar_width,self.polar_width)/self.polar_width
        pxCirclePoints = pupil_xyr[0] + pupil_xyr[2]*np.cos(theta)
        pyCirclePoints = pupil_xyr[1] + pupil_xyr[2]*np.sin(theta)
        
        ixCirclePoints = iris_xyr[0] + iris_xyr[2]*np.cos(theta)
        iyCirclePoints = iris_xyr[1] + iris_xyr[2]*np.sin(theta)

        radius = np.linspace(0,self.polar_height,self.polar_height)/self.polar_height
        for j in range(self.polar_width):
            x = (np.clip(0,width-1,np.around((1-radius) * pxCirclePoints[j] + radius * ixCirclePoints[j]))).astype(int)
            y = (np.clip(0,height-1,np.around((1-radius) * pyCirclePoints[j] + radius * iyCirclePoints[j]))).astype(int)
            
            for i in range(self.polar_height):
                if (x[i] > 0 and x[i] < width and y[i] > 0 and y[i] < height): 
                    image_polar[i][j] = image[y[i]][x[i]]
                    mask_polar[i][j] = 255*mask[y[i]][x[i]]

        return image_polar, mask_polar
    
    
    # Iris code
    def extractCode(self, polar):
        
        if polar is None:
            return None
        
        # Wrap image
        r = int(np.floor(self.filter_size / 2));
        imgWrap = np.zeros((r*2+self.polar_height, r*2+self.polar_width))
        imgWrap[:r, :r] = polar[-r:, -r:]
        imgWrap[:r, r:-r] = polar[-r:, :]
        imgWrap[:r, -r:] = polar[-r:, :r]

        imgWrap[r:-r, :r] = polar[:, -r:]
        imgWrap[r:-r, r:-r] = polar
        imgWrap[r:-r, -r:] = polar[:, :r]

        imgWrap[-r:, :r] = polar[:r, -r:]
        imgWrap[-r:, r:-r] = polar[:r, :]
        imgWrap[-r:, -r:] = polar[:r, :r]

        # Loop over all BSIF kernels in the filter set
        codeBinary = np.zeros((self.polar_height, self.polar_width, self.num_filters))
        for i in range(1,self.num_filters+1):
            ci = scipy.signal.convolve2d(imgWrap, np.rot90(self.filter[:,:,self.num_filters-i],2), mode='valid')
            codeBinary[:,:,i-1] = ci>0

        return codeBinary

    def extractRawCode(self, polar):
        
        if polar is None:
            return None
        
        # Wrap image
        r = int(np.floor(self.filter_size / 2));
        imgWrap = np.zeros((r*2+self.polar_height, r*2+self.polar_width))
        imgWrap[:r, :r] = polar[-r:, -r:]
        imgWrap[:r, r:-r] = polar[-r:, :]
        imgWrap[:r, -r:] = polar[-r:, :r]

        imgWrap[r:-r, :r] = polar[:, -r:]
        imgWrap[r:-r, r:-r] = polar
        imgWrap[r:-r, -r:] = polar[:, :r]

        imgWrap[-r:, :r] = polar[:r, -r:]
        imgWrap[-r:, r:-r] = polar[:r, :]
        imgWrap[-r:, -r:] = polar[:r, :r]

        # Loop over all BSIF kernels in the filter set
        codeBinary = np.zeros((self.polar_height, self.polar_width, self.num_filters))
        for i in range(1,self.num_filters+1):
            ci = scipy.signal.convolve2d(imgWrap, np.rot90(self.filter[:,:,self.num_filters-i],2), mode='valid')
            codeBinary[:,:,i-1] = ci

        return codeBinary

    # Match iris codes
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

    def polar(self,x,y):
        return math.hypot(x,y),math.degrees(math.atan2(y,x))

    def visualizeMatchingResult(self, code1, code2, mask1, mask2, shift, im, pupil_xyr, iris_xyr):
        
        resMask = np.zeros((self.ISO_RES[1],self.ISO_RES[0]))

        # calculate heat map
        xorCodes = np.logical_xor(self.code1, np.roll(self.code2, self.max_shift-shift, axis=1))
        andMasks = np.logical_and(self.mask1, np.roll(self.mask2, self.max_shift-shift, axis=1))

        heatMap = 1-xorCodes.astype(int)
        heatMap = np.pad(np.mean(heatMap,axis=2), pad_width=((8,8),(0,0)), mode='constant', constant_values=0)
        andMasks = np.pad(andMasks, pad_width=((8,8),(0,0)), mode='constant', constant_values=0)
        heatMap = heatMap * andMasks

        if 'single' in self.vis_mode:
            heatMap = (heatMap >= self.visMinAgreedBits / 100).astype(np.uint8)

        heatMap = np.roll(heatMap,int(self.polar_width/2),axis=1)

        for j in range(self.ISO_RES[0]):
            for i in range(self.ISO_RES[1]):
                xi = j-iris_xyr[0]
                yi = i-iris_xyr[1]
                ri = iris_xyr[2]
                xp = j-pupil_xyr[0]
                yp = i-pupil_xyr[1]
                rp = pupil_xyr[2]

                if xi**2 + yi**2 < ri**2 and xp**2 + yp**2 > rp**2:
                    rr,tt = self.polar(xi,yi)
                    tt = np.clip(np.round(self.polar_width*((180+tt)/360)).astype(int),0,self.polar_width-1)
                    rr = np.clip(np.round(self.polar_height * (rr - rp) / (ri - rp)).astype(int),0,self.polar_height-1)
                    resMask[i,j] = heatMap[rr,tt] # *** TODO correct mapping for shifted p/i centers 
        
        heatMap = 255*cv2.morphologyEx(resMask, cv2.MORPH_OPEN, kernel=self.se)
        mask_blur = cv2.filter2D(heatMap,-1,self.sk)

        if 'single' in self.vis_mode:
            mask_blur = (48 * mask_blur / np.max(mask_blur)).astype(int)
            imVis = np.stack((np.array(im),)*3, axis=-1)
            imVis[:,:,1] = np.clip(imVis[:,:,1] + mask_blur,0,255)
        elif 'heat_map' in self.vis_mode:
            mask_blur = (255 * mask_blur / np.max(mask_blur)).astype(int)
            heatMap = np.uint8(np.stack((np.array(mask_blur),)*3, axis=-1))
            heatMap = cv2.applyColorMap(heatMap, cv2.COLORMAP_JET)
            cl_im = self.clahe.apply(np.array(im))
            imVis = np.stack((cl_im,)*3, axis=-1)
            imVis = cv2.addWeighted(heatMap, 0.1, np.array(imVis), 0.9, 32)
        else:
            raise Exception("Unknown visualization mode")

        return imVis
