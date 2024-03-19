import torch
from torch import Tensor
import math
import numpy as np
import cv2

__all__ = [
    "LocalDiskBlur",
    "local_disk_blur",
]

def disk_blur_kernels(coc_size, size = 11):
    pi = math.pi
    
    coc_size = coc_size.unsqueeze(1).unsqueeze(2).repeat(1, size, size) / 2.0
    b, _, _ = coc_size.shape

        # 1. create a batch of kernels
    kernel_radius = int(size/2 - 1/2)

    kernel = torch.zeros([size, size], device = coc_size.device)
    # 2. generate the distance mask between each pixel to the center pixel
    kernel_center = float((kernel_radius))
    distances = torch.norm(torch.stack(torch.meshgrid([torch.arange(s) for s in kernel.shape]), dim=-1) - kernel_center, dim=-1)
   
    distances_batch = distances.unsqueeze(0).repeat(b, 1, 1).cuda()

    kernels = torch.where(distances_batch <= (coc_size - 0.5), torch.ones_like(distances_batch[0]), torch.zeros_like(distances_batch[0]))

    kernels = torch.where(((coc_size - 0.5) < distances_batch) * (distances_batch < (coc_size + 0.5)), (-distances_batch + coc_size + 0.5)*torch.ones_like(distances_batch[0]), kernels)

    # 3. normalize the kernel weights

    kernels = kernels/(kernels.sum(dim = [1, 2], keepdim = True)  + 1e-8)

    return kernels


def disk_blur_kernels_discrete(coc_size, size = 11):
    pi = math.pi
    
    coc_size = coc_size.unsqueeze(1).unsqueeze(2).repeat(1, size, size) #/ 2.0
    b, _, _ = coc_size.shape
    
    # 1. create a batch of kernels
    kernel_radius = int(size/2 - 1/2)


    kernel = torch.zeros([size, size], device = coc_size.device)
    # 2. generate the distance mask between each pixel to the center pixel
    kernel_center = float((kernel_radius))
    distances = torch.norm(torch.stack(torch.meshgrid([torch.arange(s) for s in kernel.shape]), dim=-1) - kernel_center, dim=-1)
 
    distances_batch = distances.unsqueeze(0).repeat(b, 1, 1).cuda()

    kernels = torch.where(distances_batch <= coc_size, torch.ones_like(distances_batch[0]), torch.zeros_like(distances_batch[0]))
  
    # 3. normalize the kernel weights
    #print('kernels sum up is', kernels.sum(dim = [1, 2], keepdim = True))
    kernels = kernels/(kernels.sum(dim = [1, 2], keepdim = True)  + 1e-8)

    return kernels

def local_disk_blur(input, modulator, kernel_size=11):
    """Blurs image with dynamic Gaussian blur.
    
    Args:
        input (Tensor): The image to be blurred (C,H,W).
        modulator (Tensor): The modulating signal that determines the local value of kernel variance (H,W).
        kernel_size (int): Size of the Gaussian kernel.

    Returns:
        Tensor: Locally blurred version of the input image.

    """       
    if len(input.shape) < 4:
        input = input.unsqueeze(0)

    # modulator = torch.where(modulator > kernel_size, kernel_size*torch.ones_like(modulator), modulator)
    #modulator = torch.where(modulator < kernel_size, 0.00000001*torch.ones_like(modulator), modulator)

    b,c,h,w = input.shape
    pad = int((kernel_size-1)/2)

    # 1. pad the input with replicated values
    inp_pad = torch.nn.functional.pad(input, pad=(pad,pad,pad,pad), mode='replicate')
    # 2. Create a Tensor of varying Gaussian Kernel
    kernels = disk_blur_kernels(modulator.flatten(), size = kernel_size).view(b,-1,kernel_size,kernel_size)  
    # print('kernels shape is', kernels.shape)  
    #kernels_rgb = torch.stack(c*[kernels], 1)
    kernels_rgb=kernels.unsqueeze(1).expand(kernels.shape[0],c,*kernels.shape[1:])
    # 3. Unfold input
    inp_unf = torch.nn.functional.unfold(inp_pad, (kernel_size,kernel_size))  
    # 4. Multiply kernel with unfolded
    x1 = inp_unf.view(b,c,-1,h*w)
    x2 = kernels_rgb.view(b,c,h*w,-1).permute(0,1,3,2)#.unsqueeze(0)
    y = (x1*x2).sum(2)
    # 5. Fold and return
    return torch.nn.functional.fold(y,(h,w),(1,1))

class LocalDiskBlur(torch.nn.Module):
    """Blurs image with dynamic Gaussian blur.
    If the image is torch Tensor, it is expected
    to have [B, C, H, W] shape.
    
    Args:
        kernel_size (int): Size of the Gaussian kernel.

    Returns:
        Tensor: Gaussian blurred version of the input image.

    """     
    
    def __init__(self, kernel_size=11):
        super().__init__()
        self.kernel_size = kernel_size


    def forward(self, img: Tensor, modulator: Tensor) -> Tensor:
        """
        Args:
            img (Tensor): image to be blurred.
            modulator (Tensor): signal modulating the kernel variance (shape H x W).

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        return local_disk_blur(img, modulator, kernel_size=self.kernel_size)


    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(kernel_size={self.kernel_size}"
        return s


