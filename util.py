import os
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def get_mgrid(sidelen, vmin=-1, vmax=1):
    if type(vmin) is not list:
        vmin = [vmin for _ in range(len(sidelen))]
    if type(vmax) is not list:
        vmax = [vmax for _ in range(len(sidelen))]
    tensors = tuple([torch.linspace(vmin[i], vmax[i], steps=sidelen[i]) for i in range(len(sidelen))])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, len(sidelen))
    return mgrid

def apply_homography(x, h):
    h = torch.cat([h, torch.ones_like(h[:, [0]])], -1)
    h = h.view(-1, 3, 3)
    x = torch.cat([x, torch.ones_like(x[:, 0]).unsqueeze(-1)], -1).unsqueeze(-1)
    o = torch.bmm(h, x).squeeze(-1)
    o = o[:, :-1] / o[:, [-1]]
    return o

def jacobian(y, x):
    B, N = y.shape
    jacobian = list()
    for i in range(N):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = torch.autograd.grad(y,
                                      x,
                                      grad_outputs=v,
                                      retain_graph=True,
                                      create_graph=True)[0]  # shape [B, N]
        jacobian.append(dy_i_dx)
    jacobian = torch.stack(jacobian, dim=1).requires_grad_()
    return jacobian


class VideoFitting(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        self.path = path
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

        self.video = self.get_video_tensor()
        self.num_frames, _, self.H, self.W = self.video.size()
        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, 3)
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

        shuffle = torch.randperm(len(self.pixels))
        self.pixels = self.pixels[shuffle]
        self.coords = self.coords[shuffle]

    def get_video_tensor(self):
        frames = sorted(os.listdir(self.path))
        video = []
        for i in range(len(frames)):
            img = Image.open(os.path.join(self.path, frames[i]))
            img = self.transform(img)
            video.append(img)
        return torch.stack(video, 0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels


class VideoFitting_depth(Dataset):
    def __init__(self, path, path_depth, transform=None, transform_depth=None):
        super().__init__()

        self.path = path
        self.path_depth = path_depth
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform
        if transform_depth is None:
            self.transform_depth = ToTensor()
        else:
            self.transform_depth = transform_depth

        self.video, self.video_depth = self.get_video_tensor()
        self.num_frames, _, self.H, self.W = self.video.size()
        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, 3)
        #print('pixels size', self.video.permute(2, 3, 0, 1).shape)
        self.pixels_depth = self.video_depth.permute(2, 3, 0, 1).contiguous().view(-1, 1)        
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

        # shuffle = torch.randperm(len(self.pixels))
        # self.pixels = self.pixels[shuffle]
        # self.pixels_depth = self.pixels_depth[shuffle]
        # self.coords = self.coords[shuffle]

    def get_video_tensor(self):
        frames = sorted(os.listdir(self.path))
        video = []
        video_depth = []
        for i in range(len(frames)):
            img = Image.open(os.path.join(self.path, frames[i]))
            print('depth path is', self.path_depth, frames[i].replace('.jpg', '.png'))
            img_depth = cv2.imread(os.path.join(self.path_depth, frames[i].replace('.jpg', '.png')), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

            img = self.transform(img)
            #print('depth size', img_depth.shape)
            img_depth = img_depth.astype(np.float32)/65535.0
            #print('depth size after', img_depth.shape)
            #print(np.max(img_depth))
            #print(np.min(img_depth))
            img_depth = self.transform_depth(img_depth)
            video.append(img)
            video_depth.append(img_depth)
        return torch.stack(video, 0), torch.stack(video_depth, 0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels, self.pixels_depth

class VideoFitting_noise(Dataset):
    def __init__(self, path,  transform=None):
        super().__init__()

        self.path = path
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform
        # if transform_depth is None:
        #     self.transform_depth = ToTensor()
        # else:
        #     self.transform_depth = transform_depth

        self.video = self.get_video_tensor()
        self.num_frames, _, self.H, self.W = self.video.size()
        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, 3)
        #print('pixels size', self.video.permute(2, 3, 0, 1).shape)
        #self.pixels_depth = self.video_depth.permute(2, 3, 0, 1).contiguous().view(-1, 1)        
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

        # shuffle = torch.randperm(len(self.pixels))
        # self.pixels = self.pixels[shuffle]
        # self.pixels_depth = self.pixels_depth[shuffle]
        # self.coords = self.coords[shuffle]

    def get_video_tensor(self):
        frames = sorted(os.listdir(self.path))
        video = []
        for i in range(len(frames)):
            img = Image.open(os.path.join(self.path, frames[i]))
            print('path is', os.path.join(self.path, frames[i]))
            # img_depth = cv2.imread(os.path.join(self.path_depth, frames[i].replace('.JPG', '.png')), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

            img = self.transform(img)
            # print('depth size', img_depth.shape)
            # img_depth = cv2.resize(img_depth, [195, 130]).astype(np.float32)/65535.0
            # print('depth size after', img_depth.shape)
            # print(np.max(img_depth))
            # print(np.min(img_depth))
            # img_depth = self.transform_depth(img_depth)
            video.append(img)
            # video_depth.append(img_depth)
        return torch.stack(video, 0)#, torch.stack(video_depth, 0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels

class VideoFitting_Smartphone(Dataset):
    def __init__(self, path,  transform=None):
        super().__init__()

        self.path = path
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform
        # if transform_depth is None:
        #     self.transform_depth = ToTensor()
        # else:
        #     self.transform_depth = transform_depth

        self.video = self.get_video_tensor()
        self.num_frames, _, self.H, self.W = self.video.size()
        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, 3)
        #print('pixels size', self.video.permute(2, 3, 0, 1).shape)
        #self.pixels_depth = self.video_depth.permute(2, 3, 0, 1).contiguous().view(-1, 1)        
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

        # shuffle = torch.randperm(len(self.pixels))
        # self.pixels = self.pixels[shuffle]
        # self.pixels_depth = self.pixels_depth[shuffle]
        # self.coords = self.coords[shuffle]

    def get_video_tensor(self):
        frames = sorted(os.listdir(self.path))
        video = []
        for i in range(len(frames)):
            print('frames[i] is', frames[i])
            img = cv2.cvtColor(cv2.imread(os.path.join(self.path, frames[i])), cv2.COLOR_BGR2RGB).astype(np.float32)/255.#Image.open(os.path.join(self.path, frames[i]))
            img = img[12:-12, 13:-13, :]
            # img_depth = cv2.imread(os.path.join(self.path_depth, frames[i].replace('.JPG', '.png')), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

            img = self.transform(img)
            # print('depth size', img_depth.shape)
            # img_depth = cv2.resize(img_depth, [195, 130]).astype(np.float32)/65535.0
            # print('depth size after', img_depth.shape)
            # print(np.max(img_depth))
            # print(np.min(img_depth))
            # img_depth = self.transform_depth(img_depth)
            video.append(img)
            # video_depth.append(img_depth)
        return torch.stack(video, 0)#, torch.stack(video_depth, 0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels


class VideoFitting_cube(Dataset):
    def __init__(self, path, name_list, transform=None):
        super().__init__()

        self.path = path
        self.name_list = name_list
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

        self.video = self.get_video_tensor()
        self.num_frames, _, self.H, self.W = self.video.size()
        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, 3)     
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

    def get_video_tensor(self):
        # frames = sorted(os.listdir(self.path))
        video = []
        for i in range(len(self.name_list)):
            img = Image.open(os.path.join(self.path, self.name_list[i]))
            print('path is', os.path.join(self.path, self.name_list[i]))

            img = self.transform(img)
            video.append(img)

        return torch.stack(video, 0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
        return self.coords, self.pixels