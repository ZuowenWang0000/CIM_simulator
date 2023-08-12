import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import numbers
import torchvision
from torchvision import transforms
import cv2 as cv


def convlayer(n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
    layer = [
        nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(n_output),
        nn.ReLU(inplace=True),
        resample_out]
    if resample_out is None:
        layer.pop()
    return layer  

class convBlock(nn.Module):
    def __init__(self, n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
        super(convBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(n_output)
        self.relu1 = nn.ReLU(inplace=True)
        self.resample_out = resample_out
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        if self.resample_out is not None:
            out = self.resample_out(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, stride=1, resample_out=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.resample_out = resample_out
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # out += residual
        out = self.relu2(out)
        if self.resample_out is not None:
            out = self.resample_out(out)
        return out

class VGG_Feature_Extractor(object):
    def __init__(self,layer_depth=4,device='cuda:0'):
        """Use the first <layer_depth> layers of the vgg16 model pretrained on imageNet as feature extractor.  
        When called, returns the feature maps of input image. (grayscale input is automatically converted to RGB"""
        model = torchvision.models.vgg16(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[0][:int(layer_depth)]).to(device)
        for child in [*self.feature_extractor]:
            for p in child.parameters():
                p.requires_grad = False
        self.rgbConverter = RGBConverter()
    def __call__(self,image):
        if image.shape[1]==1:
            image = self.rgbConverter(image)
            return self.feature_extractor(image)
        else:
            assert image.shape[1]==3
            return self.feature_extractor(image)    


class Quantized_E2E_Encoder(nn.Module):
    def __init__(self, model_fp32):
        super(Quantized_E2E_Encoder, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        # print("--------------")
        # print(self.dequant)
        # print(self.dequant.qconfig)
        # print("--------------")

        return x


class E2E_Encoder(nn.Module):
    """
    Simple non-generic encoder class that receives 128x128 input and outputs 32x32 feature map as stimulation protocol
    """   
    def __init__(self, in_channels=3, out_channels=1):
        super(E2E_Encoder, self).__init__()
             
        # self.binary_stimulation = binary_stimulation
        
        # # Model
        self.model = nn.Sequential(convBlock(in_channels,8,3,2,1),
                                   convBlock(8,16,3,1,1,resample_out=None),
                                   convBlock(16,28,3,2,1,resample_out=None),
                                   convBlock(28,28,3,1,1,resample_out=None),
                                   ResidualBlock(28, resample_out=None),
                                   ResidualBlock(28, resample_out=None),
                                   ResidualBlock(28, resample_out=None),
                                   ResidualBlock(28, resample_out=None),
                                   convBlock(28,16,3,1,1),
                                   nn.Conv2d(16,out_channels,3,1,1))
                                #    nn.Tanh())
                                #    nn.ReLU())
    def forward(self, x):
        self.out = self.model(x)
        
        
        self.out = torch.tanh(self.out)
        return self.out

        # x = self.out
        # if self.binary_stimulation:
        #     x = x + torch.sign(x).detach() - x.detach() # (self-through estimator)
        # stimulation = .5*(x+1)
        # return stimulation    
    
class E2E_Decoder(nn.Module):
    """
    Simple non-generic phosphene decoder.
    in: (256x256) SVP representation
    out: (128x128) Reconstruction
    """   
    def __init__(self, in_channels=1, out_channels=1, out_activation='sigmoid'):
        super(E2E_Decoder, self).__init__()
             
        # Activation of output layer
        self.out_activation = {'tanh': nn.Tanh(),
                               'sigmoid': nn.Sigmoid(),
                               'relu': nn.LeakyReLU(),
                               'softmax':nn.Softmax(dim=1)}[out_activation]
        
        # Model
        self.model = nn.Sequential(*convlayer(in_channels,16,3,1,1),
                                   *convlayer(16,32,3,1,1),
                                   *convlayer(32,64,3,2,1),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   ResidualBlock(64),
                                   *convlayer(64,32,3,1,1),
                                   nn.Conv2d(32,out_channels,3,1,1), 
                                   self.out_activation)       

    def forward(self, x):
        return self.model(x)    
    
    
class E2E_PhospheneSimulator(nn.Module):
    """ Uses three steps to convert  the stimulation vectors to phosphene representation:
    1. Resizes the feature map (default: 32x32) to SVP template (256x256)
    2. Uses pMask to sample the phosphene locations from the SVP activation template
    2. Performs convolution with gaussian kernel for realistic phosphene simulations
    """
    def __init__(self,pMask,scale_factor=8, sigma=1.5,kernel_size=11, intensity=15, device=torch.device('cuda:0'), binary_stimulation=True):
        super(E2E_PhospheneSimulator, self).__init__()
        
        # Device
        self.device = device
        self.binary_stimulation = binary_stimulation
        
        # Phosphene grid
        self.pMask = pMask
        self.up = nn.Upsample(mode="nearest",scale_factor=scale_factor)
        self.gaussian = self.get_gaussian_layer(kernel_size=kernel_size, sigma=sigma, channels=1)
        self.intensity = intensity 
    
    def get_gaussian_layer(self, kernel_size, sigma, channels):
        """non-trainable Gaussian filter layer for more realistic phosphene simulation"""

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) /\
                              (2*variance)
                          )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter    

    def forward(self, stimulation):

        # ## stimulation
        # print(stimulation.shape)
        # print("----------------------------")
        # from torchvision.utils import save_image
        # for stim_i in range(stimulation.shape[0]):
        #     stim = stimulation[stim_i]
        #     # print(stim.shape)
        #     # print(stim)
        #     # exit() 
        #     string = 'stimulation_images/stim_before_binary_stimulation_' + str(stim_i) +'.png'
        #     save_image(stim, string)
        # # exit()
        
        # stimulation = torch.tanh(stimulation)
        if self.binary_stimulation:
            stimulation = stimulation + torch.sign(stimulation).detach() - stimulation.detach() # (self-through estimator)
        stimulation = .5*(stimulation+1)

        # for stim_i in range(stimulation.shape[0]):
        #     stim = stimulation[stim_i] 
        #     # img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
        #     string = 'stimulation_images/stim_after_binary_stimulation_' + str(stim_i) +'.png'
        #     save_image(stim, string)
        # exit()
        
        # Phosphene simulation
        phosphenes = self.up(stimulation)*self.pMask
        phosphenes = self.gaussian(F.pad(phosphenes, (5,5,5,5), mode='constant', value=0)) 
        return self.intensity*phosphenes    

class E2E_CannyModel(nn.Module):
    """Uses openCVs Canny edge detection module for image filtering. 
    The edge map is converted to a stimulation map (by downsampling to n_phosphenes*n_phosphenes"""
    def __init__(self,scale_factor,device,imsize=(128,128),ksize=(7,7),sigma=1,low=50,high=100):
        super(E2E_CannyModel, self).__init__()
        
        self.device = device        
        self.to_cv2_list = lambda image_tensor : [np.squeeze((255*img.cpu().numpy())).astype('uint8') for img in image_tensor]
        self.gaus_blur   = lambda image_list : [cv.GaussianBlur(img,ksize=ksize,sigmaX=sigma) for img in image_list]
        self.canny_edge  = lambda image_list : [cv.Canny(img,low,high) for img in image_list]
        self.to_tensor   = lambda image_list : torch.tensor(image_list, device=device,dtype=torch.float32).unsqueeze(axis=1)
        self.interpolate = lambda image_tensor : F.interpolate(image_tensor,scale_factor=scale_factor)
        
        self.model = transforms.Compose([transforms.Lambda(self.to_cv2_list),
                                             transforms.Lambda(self.gaus_blur),
                                             transforms.Lambda(self.canny_edge),
                                             transforms.Lambda(self.to_tensor),
                                             transforms.Lambda(self.interpolate)])
                                                   
    def forward(self, x):
        return  self.model(x)/255  
    
