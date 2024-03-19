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

from model import Siren, TMO
from util import get_mgrid, jacobian, VideoFitting_noise
import imageio
import cv2
import math
import torch.nn.functional as F
import scipy.stats as st
from src.Localdiskblur_parallel_modified import *
from torchvision.models import vgg16
import json
import argparse
from torch.utils.tensorboard import SummaryWriter

def coc_blur(focal_distance, pixel_depth, focal_length, aperture, pixel_size):
    s1 = focal_distance
    s2 = pixel_depth

    s2 = torch.exp(0.5*s2)*1000.0

    coc_diameter = torch.abs((s2 - s1))/s2*(focal_length**2)/(aperture*(s1 - focal_length))

    coc_diameter_pixel = coc_diameter/16/pixel_size

    bottom = (torch.ones([1])).cuda()

    coc_diameter_pixel = torch.where(coc_diameter_pixel < bottom, bottom , coc_diameter_pixel)

    return coc_diameter_pixel

class TVLoss(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
 
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()  

        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class MultiscaleVGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(MultiscaleVGGPerceptualLoss, self).__init__()

        vgg = vgg16(pretrained=True)

        self.feature_extractor_1 = torch.nn.Sequential(*list(vgg.features)[:5]).eval()
        self.feature_extractor_2 = torch.nn.Sequential(*list(vgg.features)[:10]).eval()
        self.feature_extractor_3 = torch.nn.Sequential(*list(vgg.features)[:16]).eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):

        x_features_1 = self.feature_extractor_1(x)
        x_features_2 = self.feature_extractor_2(x)
        x_features_3 = self.feature_extractor_3(x)

        y_features_1 = self.feature_extractor_1(y)
        y_features_2 = self.feature_extractor_2(y)
        y_features_3 = self.feature_extractor_3(y)


        mse = torch.nn.MSELoss()
        loss_1 = mse(x_features_1, y_features_1)
        loss_2 = mse(x_features_2, y_features_2)
        loss_3 = mse(x_features_3, y_features_3)


        loss = (loss_1 + loss_2 + loss_3)/3.0
        return loss

def train_defocus(path, para_path, save_path, total_steps, verbose=True, steps_til_summary=10):
    transform = Compose([
        ToTensor(),
        Resize([256, 384]),
        Normalize(torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5]))
    ])

    v = VideoFitting_noise(path, transform)
    videoloader = DataLoader(v, batch_size=1, pin_memory=True, num_workers=0)

    with open(para_path + 'parameters.json', 'r') as file:
        config = json.load(file)
    focus_distance_list_tmp = config['focal_distance']
    focus_distance_list_np = np.array(focus_distance_list_tmp)*1000.0
    focal_length = config['focal_length']*1000.0
    aperture_list = config['aperture']
    expsoure_list_np = config['exposure']
    if 'pixel_size' not in config:
        pixel_size = 0.0058
    else:
        pixel_size = config['pixel_size']

    writer = SummaryWriter('%s/runs'%(save_path))
    
    ##define the MLPs: all-infocus, depth, flow, and TMO
    f = Siren(in_features=2, out_features=3, hidden_features=256, 
              hidden_layers=4, outermost_linear=True)

    depth_predictor = Siren(in_features=2, out_features=1, hidden_features=256, hidden_layers=4, outermost_linear=True)
    xy_offset_mlp = Siren(in_features=3, out_features=2, hidden_features=256, hidden_layers=2, outermost_linear=True)
    tmo_mlp = TMO()
    f.cuda()
    depth_predictor.cuda()
    xy_offset_mlp.cuda()
    tmo_mlp.cuda()
    lblur = LocalDiskBlur(kernel_size = 21)
    loss_TV = TVLoss(TVLoss_weight = 0.05)
    vgg_loss = MultiscaleVGGPerceptualLoss().cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    focus_distance_list = torch.tensor(focus_distance_list_np, requires_grad=True, device = device)

    expsoure_list_middle = torch.tensor(expsoure_list_np, requires_grad=True, device = device) 

    params = list(f.parameters()) + list(depth_predictor.parameters()) + [focus_distance_list] + list(xy_offset_mlp.parameters()) + [expsoure_list_middle] + list(tmo_mlp.parameters())
    optim = torch.optim.Adam(params, lr=1e-4)
    print('depth params are')

    t = depth_predictor.to(device)
    print('model is', t)

    model_input, ground_truth = next(iter(videoloader))
    model_input, ground_truth = model_input[0].cuda(), ground_truth[0].cuda()

    batch_size = (v.H * v.W) #// 4

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path_edit = save_path + '/focal_stack_edit/'
    if not os.path.exists(save_path_edit):
        os.makedirs(save_path_edit)
    aperture_list = aperture_list 

    for step in range(total_steps):
        for m in range(v.num_frames):
            model_input_select = model_input[m:len(model_input):v.num_frames, :]
            ground_truth_select = ground_truth[m:len(model_input):v.num_frames, :]

            focus_distance = focus_distance_list[m]
            aperture = aperture_list[m]

            model_input_offset = torch.cat((model_input_select[:, :-1], (focus_distance/1000.0).repeat([batch_size, 1])), axis=1).requires_grad_()

            offset = xy_offset_mlp(model_input_offset.float())
            all_in_focus = f(model_input_select[:, :-1] + offset)
            all_in_focus = torch.exp(all_in_focus)
            depth_select = depth_predictor(model_input_select[:, :-1] + offset)
             
            all_in_focus_img = all_in_focus.contiguous().view(v.H, v.W, -1, 3)
            all_in_focus_img_permute = all_in_focus_img.permute(2, 3, 0, 1)
            coc_size = coc_blur(focal_distance = focus_distance, focal_length = focal_length, pixel_depth = depth_select, aperture = aperture, pixel_size = pixel_size)
            coc_size_permute = coc_size.contiguous().view(v.H, v.W, 1, 1).permute(2, 3, 0, 1)
            blur_img = lblur(all_in_focus_img_permute, coc_size_permute)
            expsoure = expsoure_list_middle[m]  
            expsoure = expsoure * torch.ones_like(blur_img).cuda()
            #TMO
            blur_img_log_exp = blur_img*2**expsoure
            blur_img_log_exp_view = blur_img_log_exp.permute(2, 3, 0, 1).view(-1, 3)
            #split into RGB
            blur_img_log_exp_r = blur_img_log_exp_view[:, 0:1]
            blur_img_log_exp_g = blur_img_log_exp_view[:, 1:2]
            blur_img_log_exp_b = blur_img_log_exp_view[:, 2:3]

                        
            blur_img_tmo_r = tmo_mlp(blur_img_log_exp_r)
            blur_img_tmo_g = tmo_mlp(blur_img_log_exp_g)
            blur_img_tmo_b = tmo_mlp(blur_img_log_exp_b)

            blur_img_tmo = torch.cat([blur_img_tmo_r, blur_img_tmo_g, blur_img_tmo_b], -1)  
            
            ground_truth_select_img = ground_truth_select.contiguous().view(v.H, v.W, -1, 3).permute(2, 3, 0, 1)
            
            loss =  0.01*vgg_loss(ground_truth_select_img, blur_img_tmo.contiguous().view(v.H, v.W, -1, 3).permute(2, 3, 0, 1)) \
                    + 0.5*jacobian(offset, model_input_offset).abs().mean() \
                    + ((blur_img_tmo_r - ground_truth_select[:, 0:1]) ** 2).mean() + ((blur_img_tmo_g - ground_truth_select[:, 1:2]) ** 2).mean() + ((blur_img_tmo_b - ground_truth_select[:, 2:3]) ** 2).mean()\
                    + loss_TV(depth_select.view(v.H, v.W, -1, 1).permute(2, 3, 0, 1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            if (m == v.num_frames - 1) and not step % steps_til_summary:
                writer.add_scalar("loss", loss, step)
            if verbose and not step % steps_til_summary:
                print("Step [%04d/%04d]: loss=%0.4f" % (step, total_steps, loss))

                if m == (v.num_frames - 1):

                    with torch.no_grad():
                        for j in range(v.num_frames):
                            model_input_select = model_input[j:len(model_input):v.num_frames, :]
                            ground_truth_select = ground_truth[j:len(model_input):v.num_frames, :]
                            
                            model_input_offset = torch.cat((model_input_select[:, :-1], (focus_distance_list[j]/1000.0).repeat([batch_size, 1])), axis=1)
                            offset = xy_offset_mlp(model_input_offset.float())
                            depth_select_vis = depth_predictor(model_input_select[:, :-1] + offset)
                            all_in_focus_vis = f(model_input_select[:, :-1] + offset)
                            all_in_focus_vis = torch.exp(all_in_focus_vis)
                            all_in_focus_vis_img = all_in_focus_vis.contiguous().view(v.H, v.W, -1, 3)
                            all_in_focus_vis_img_permute = all_in_focus_vis_img.permute(2, 3, 0, 1)
                            coc_size_t = coc_blur(focal_distance = focus_distance_list[j], focal_length = focal_length, pixel_depth = depth_select_vis, aperture = aperture_list[j], pixel_size = pixel_size)                            
                            blur_img = lblur(all_in_focus_vis_img_permute, coc_size_t.contiguous().view(v.H, v.W, 1).permute(2, 0, 1))
                            
                            blur_img_log_exp = blur_img * 2**expsoure_list_middle[j]
                            
                            ## split into RGB
                            blur_img_log_exp_view = blur_img_log_exp.permute(2, 3, 0, 1).view(-1, 3)
                            blur_img_log_exp_r = blur_img_log_exp_view[:, 0:1]
                            blur_img_log_exp_g = blur_img_log_exp_view[:, 1:2]
                            blur_img_log_exp_b = blur_img_log_exp_view[:, 2:3]

                            blur_img_tmo_r = tmo_mlp(blur_img_log_exp_r)
                            blur_img_tmo_g = tmo_mlp(blur_img_log_exp_g)
                            blur_img_tmo_b = tmo_mlp(blur_img_log_exp_b)

                            blur_img_tmo = torch.cat([blur_img_tmo_r, blur_img_tmo_g, blur_img_tmo_b], -1)                            
                            blur_img_numpy = (blur_img_tmo).view(v.H, v.W, 3).cpu().detach().numpy()
                            blur_img_numpy_out = np.clip(blur_img_numpy, -1, 1) * 0.5 + 0.5 

                            imageio.imsave('%s/output_%s_%s.png'%(save_path, step, j), blur_img_numpy_out)
                            
                            all_in_focus_vis_img_permute_numpy = all_in_focus_vis.view(v.H, v.W, 3).cpu().detach().numpy()

                            cv2.imwrite('%s/all_in_focus_image_%s_%s.hdr'%(save_path, step, j), cv2.cvtColor(all_in_focus_vis_img_permute_numpy, cv2.COLOR_RGB2BGR))
 
                            depth_opt_permute_numpy = torch.exp(0.5*depth_select_vis).view(v.H, v.W, 1).cpu().detach().numpy()


                        model_input_all = get_mgrid([(v.H + 20), (v.W + 20)], vmin = -1.2, vmax = 1.2).cuda()
                        depth_select_vis_all = depth_predictor(model_input_all) 
                        all_in_focus_vis_all = f(model_input_all)
                        all_in_focus_vis_all = torch.exp(all_in_focus_vis_all)
                        all_in_focus_vis_img_permute_all_numpy = all_in_focus_vis_all.view(v.H + 20, v.W + 20, 3).cpu().detach().numpy()
                        cv2.imwrite('%s/all_in_focus_image_all_%s.hdr'%(save_path, step), cv2.cvtColor(all_in_focus_vis_img_permute_all_numpy, cv2.COLOR_RGB2BGR))
                        depth_opt_permute_all_numpy = torch.exp(0.5*depth_select_vis_all).view(v.H+20, v.W+20, 1).cpu().detach().numpy()
                        cv2.imwrite('%s/depth_map_all_%s.hdr'%(save_path, step), depth_opt_permute_all_numpy)



    torch.save(f.state_dict(), '%s/model_color_final.pkl'%(save_path))
    torch.save(depth_predictor.state_dict(), '%s/model_depth_final.pkl'%(save_path))
    torch.save(tmo_mlp.state_dict(), '%s/model_tmo_final.pkl'%(save_path))
    torch.save(xy_offset_mlp.state_dict(), '%s/model_flow_final.pkl'%(save_path))  
    return f, depth_select, ground_truth


parser = argparse.ArgumentParser(description='Load configuration parameters.')
parser.add_argument('--save_path', type=str, help='save_path')
parser.add_argument('--base_path', type=str, help='base folder of the test images and parameters')
args = parser.parse_args()

save_path = args.save_path
base_path = args.base_path
rgb_path = base_path + 'rgb/'
para_path = base_path + 'para/'

f, depth_gt, ground_truth = train_defocus( path = rgb_path,  para_path = para_path, save_path = save_path, total_steps = 2001)

