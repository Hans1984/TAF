#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
from itertools import chain
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from model import Siren,  TMO
from util import get_mgrid, jacobian, VideoFitting_noise
import imageio
import cv2
import math
import torch.nn.functional as F
import scipy.stats as st
from src.Localdiskblur_parallel_modified import *
from torchvision.models import vgg16


def coc_blur(focal_distance, pixel_depth, focal_length = 85, aperture = 4, pixel_size = 0.0058):
    s1 = focal_distance
    s2 = pixel_depth
    s2 = torch.exp(0.5*s2)*1000.0

    coc_diameter = torch.abs((s2 - s1))/s2*(focal_length**2)/(aperture*(s1 - focal_length))

    coc_diameter_pixel = coc_diameter/9/pixel_size
    bottom = (torch.ones([1])).cuda()
    coc_diameter_pixel = torch.where(coc_diameter_pixel < bottom, bottom , coc_diameter_pixel)

    return coc_diameter_pixel


## load the weights
f_path = './weights/model_color_2000.pkl'
depth_path = './weights/model_depth_2000.pkl'
flow_path = './weights/model_flow_2000.pkl'
tmo_path = './weights/model_tmo_2000.pkl'
f = Siren(in_features=2, out_features=3, hidden_features=256, 
          hidden_layers=4, outermost_linear=True).cuda()#ReLU_PE_Model_modified([2,256,256,256,256,256,3], L=10)#Siren(in_features=2, out_features=3, hidden_features=256, 
          # hidden_layers=4, outermost_linear=True)
depth_predictor = Siren(in_features=2, out_features=1, hidden_features=256, hidden_layers=4, outermost_linear=True).cuda()#ReLU_PE_Model([2,256,256,256,256,1], L=10)#Depth_predictor(in_features=2, hidden_features=256, hidden_layers=4)#Siren(in_features=2, out_features=1, hidden_features=256, hidden_layers=4, outermost_linear=True)
xy_offset_mlp = Siren(in_features=3, out_features=2, hidden_features=256, hidden_layers=2, outermost_linear=True).cuda()
tmo_mlp = TMO().cuda()

state_dict_load_f = torch.load(f_path)
f.load_state_dict(state_dict_load_f)
state_dict_load_depth = torch.load(depth_path)
depth_predictor.load_state_dict(state_dict_load_depth)
state_dict_load_flow = torch.load(flow_path)
xy_offset_mlp.load_state_dict(state_dict_load_flow)
state_dict_load_tmo = torch.load(tmo_path)
tmo_mlp.load_state_dict(state_dict_load_tmo)

lblur = LocalDiskBlur(kernel_size = 47)
save_path = '/HPS/BRDF/work/chao/teaser_8_large_kernel/focal_stack_edit/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
focus_distance_list_np = np.array([0.88*1000.0, 1.17*1000.0, 1.35*1000.0, 1.48*1000.0, 2.25*1000.0])
aperture_list = np.array([2.8, 4.0, 5.6, 8.0, 11.0])
expsoure_list_np = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
focus_distance_list = torch.tensor(focus_distance_list_np).cuda()

height = 512
width = 768
with torch.no_grad():
    for j in range(len(focus_distance_list_np)):

        focus_distance = focus_distance_list[j] 
        model_input = get_mgrid([height, width, 1]).cuda()
        model_input_offset = torch.cat((model_input[:, :-1], (focus_distance_list[0]/1000.0).repeat([height*width, 1])), axis=1)
        offset = xy_offset_mlp(model_input_offset.float())
        depth_select_vis = depth_predictor(model_input[:, :-1] + offset)

        all_in_focus_vis = f(model_input[:, :-1] + offset)
        all_in_focus_vis = torch.exp(all_in_focus_vis)
        all_in_focus_vis_img = all_in_focus_vis.contiguous().view(height, width, -1, 3)
        all_in_focus_vis_img_permute = all_in_focus_vis_img.permute(2, 3, 0, 1)
        coc_size_t = coc_blur(focal_distance = focus_distance, pixel_depth = depth_select_vis, aperture = aperture_list[j])     
                     
        blur_img = lblur(all_in_focus_vis_img_permute, coc_size_t.contiguous().view(height, width, 1).permute(2, 0, 1))

        blur_img_numpy = blur_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        
        blur_img_log_exp = blur_img * 2**expsoure_list_np[j]
        
        ## split into RGB
        blur_img_log_exp_view = blur_img_log_exp.permute(2, 3, 0, 1).view(-1, 3)
        blur_img_log_exp_r = blur_img_log_exp_view[:, 0:1]
        blur_img_log_exp_g = blur_img_log_exp_view[:, 1:2]
        blur_img_log_exp_b = blur_img_log_exp_view[:, 2:3]

        blur_img_tmo_r = tmo_mlp(blur_img_log_exp_r)
        blur_img_tmo_g = tmo_mlp(blur_img_log_exp_g)
        blur_img_tmo_b = tmo_mlp(blur_img_log_exp_b)

        blur_img_tmo = torch.cat([blur_img_tmo_r, blur_img_tmo_g, blur_img_tmo_b], -1)                            
        blur_img_numpy = (blur_img_tmo).view(height, width, 3).cpu().detach().numpy()
        blur_img_numpy_out = np.clip(blur_img_numpy, -1, 1) * 0.5 + 0.5
        imageio.imsave('%s/output_%s.png'%(save_path, j), blur_img_numpy_out)