import tensorflow as tf  # version 2.5
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import ReLU,Dense, Flatten, Conv2D, Dropout, MaxPooling2D, InputLayer, ZeroPadding2D, BatchNormalization, Layer
# from tensorflow.python.keras.module import ModuleWrapper
from visualkeras.layered import layered_view
from visualkeras.layer_utils import SpacingDummyLayer
# import torch.nn as nn
import numpy as np
import math
import numbers
# import torchvision
# from torchvision import transforms
# import cv2 as cv
from collections import defaultdict
from PIL import ImageFont


def convlayer_keras(model, n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
    model.add(Conv2D(filters=n_output, kernel_size=k_size, strides=stride, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    return model

# def convlayer(n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
#     layer = [
#         nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
#         nn.BatchNorm2d(n_output),
#         nn.ReLU(inplace=True),
#         resample_out]
#     if resample_out is None:
#         layer.pop()
#     return layer  

# class convBlock(nn.Module):
#     def __init__(self, n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
#         super(convBlock, self).__init__()
#         self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)
#         self.bn1 = nn.BatchNorm2d(n_output)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.resample_out = resample_out
        
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu1(out)
#         if self.resample_out is not None:
#             out = self.resample_out(out)
#         return out

def convBlock_keras(model, n_input, n_output, k_size=3, stride=1, padding=1, resample_out=None):
    model.add(Conv2D(filters=n_output, kernel_size=k_size, strides=stride, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    if resample_out is not None:
        raise NotImplementedError
    return model

def ResidualBlock_keras(model, n_channels, stride=1, resample_out=None):
    model.add(Conv2D(filters=n_channels, kernel_size=3, strides=stride, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2D(filters=n_channels, kernel_size=3, strides=stride, padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(ReLU())
    return model

# class ResidualBlock(nn.Module):
#     def __init__(self, n_channels, stride=1, resample_out=None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
#         self.bn1 = nn.BatchNorm2d(n_channels)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1)
#         self.bn2 = nn.BatchNorm2d(n_channels)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.resample_out = resample_out
        
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu1(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         # out += residual
#         out = self.relu2(out)
#         if self.resample_out is not None:
#             out = self.resample_out(out)
#         return out
spacing = 30

image_size=224
model=Sequential()
model.add(InputLayer(input_shape=(image_size, image_size, 3)))
model = convBlock_keras(model, 3, 8, k_size=3, stride=2, padding='same', resample_out=None)
model = convBlock_keras(model, 8, 16, k_size=3, stride=1, padding='same', resample_out=None)
model = convBlock_keras(model, 16, 32, k_size=3, stride=2, padding='same', resample_out=None)
model = convBlock_keras(model, 32, 32, k_size=3, stride=1, padding='same', resample_out=None)
model.add(SpacingDummyLayer(spacing=spacing))
model = ResidualBlock_keras(model, 32, stride=1, resample_out=None)
model.add(SpacingDummyLayer(spacing=spacing))
model = ResidualBlock_keras(model, 32, stride=1, resample_out=None)
model.add(SpacingDummyLayer(spacing=spacing))
model = ResidualBlock_keras(model, 32, stride=1, resample_out=None)
model.add(SpacingDummyLayer(spacing=spacing))
model = ResidualBlock_keras(model, 32, stride=1, resample_out=None)
model.add(SpacingDummyLayer(spacing=spacing))
model = convBlock_keras(model, 32, 16, k_size=3, stride=1, padding='same', resample_out=None)
model.add(Conv2D(filters=1, kernel_size=3, strides=1, padding='same'))

color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = 'red'
color_map[BatchNormalization]['fill'] = 'yellow'
color_map[Dropout]['fill'] = 'pink'
color_map[MaxPooling2D]['fill'] = 'red'
color_map[Dense]['fill'] = 'green'
color_map[Flatten]['fill'] = 'teal'
font = ImageFont.truetype("arial.ttf", 32)

layered_view(model, to_file='./figures/encoder.png',color_map=color_map, type_ignore=[SpacingDummyLayer])
layered_view(model, to_file='./figures/encoder_legend.png',
                         legend=True, color_map=color_map, font=font)
layered_view(model, to_file='./figures/encoder_legend_volume.png', type_ignore=[SpacingDummyLayer],
                         legend=True,color_map=color_map, font=font, draw_volume=True)
layered_view(model, to_file='./figures/encoder_spacing_layers.png', color_map=color_map,spacing=5)
layered_view(model, to_file='./figures/encoder_type_ignore.png', color_map=color_map,
                         type_ignore=[Flatten, SpacingDummyLayer])
layered_view(model, to_file='./figures/encoder_color_map.png',
                         color_map=color_map, type_ignore=[SpacingDummyLayer])
layered_view(model, to_file='./figures/encoder_flat.png', color_map=color_map,
                         draw_volume=False, type_ignore=[SpacingDummyLayer])
layered_view(model, to_file='./figures/encoder_scaling.png', color_map=color_map,
                         scale_xy=1, scale_z=1, max_z=1000, type_ignore=[SpacingDummyLayer])


image_size=256
out_channels=3
model=Sequential()
model.add(InputLayer(input_shape=(image_size, image_size, 1)))
model = convlayer_keras(model, n_input=1, n_output=16, k_size=3, stride=1, padding=1, resample_out=None)
model = convlayer_keras(model, n_input=16, n_output=32, k_size=3, stride=1, padding=1, resample_out=None)
model = convlayer_keras(model, n_input=32, n_output=64, k_size=3, stride=2, padding=1, resample_out=None)
model.add(SpacingDummyLayer(spacing=spacing))
model = ResidualBlock_keras(model, 64, stride=1, resample_out=None)
model.add(SpacingDummyLayer(spacing=spacing))
model = ResidualBlock_keras(model, 64, stride=1, resample_out=None)
model.add(SpacingDummyLayer(spacing=spacing))
model = ResidualBlock_keras(model, 64, stride=1, resample_out=None)
model.add(SpacingDummyLayer(spacing=spacing))
model = ResidualBlock_keras(model, 64, stride=1, resample_out=None)
model.add(SpacingDummyLayer(spacing=spacing))
model = convlayer_keras(model, n_input=64, n_output=32, k_size=3, stride=1, padding=1, resample_out=None)
model.add(Conv2D(filters=out_channels, kernel_size=3, strides=1, padding='same'))
model.add(ReLU())
print(model.summary())

layered_view(model, to_file='./figures/decoder.png',color_map=color_map, type_ignore=[SpacingDummyLayer])
layered_view(model, to_file='./figures/decoder_legend.png', type_ignore=[SpacingDummyLayer],
                         legend=True, color_map=color_map,font=font)
layered_view(model, to_file='./figures/decoder_legend_volume.png', type_ignore=[SpacingDummyLayer],
                         legend=True, font=font, color_map=color_map,draw_volume=True)
layered_view(model, to_file='./figures/decoder_spacing_layers.png', color_map=color_map,spacing=5)
layered_view(model, to_file='./figures/decoder_type_ignore.png', color_map=color_map,
                         type_ignore=[Flatten, SpacingDummyLayer])
layered_view(model, to_file='./figures/decoder_color_map.png',
                         color_map=color_map, type_ignore=[SpacingDummyLayer])
layered_view(model, to_file='./figures/decoder_flat.png', color_map=color_map,
                         draw_volume=False, type_ignore=[SpacingDummyLayer])
layered_view(model, to_file='./figures/decoder_scaling.png', color_map=color_map,
                         scale_xy=1, scale_z=1, max_z=1000, type_ignore=[SpacingDummyLayer])

print(model.summary())
# class E2E_Decoder(nn.Module):
#     """
#     Simple non-generic phosphene decoder.
#     in: (256x256) SVP representation
#     out: (128x128) Reconstruction
#     """   
#     def __init__(self, in_channels=1, out_channels=1, out_activation='sigmoid'):
#         super(E2E_Decoder, self).__init__()
             
#         # Activation of output layer
#         self.out_activation = {'tanh': nn.Tanh(),
#                                'sigmoid': nn.Sigmoid(),
#                                'relu': nn.LeakyReLU(),
#                                'softmax':nn.Softmax(dim=1)}[out_activation]
        
#         # Model
#         self.model = nn.Sequential(*convlayer(in_channels,16,3,1,1),
#                                    *convlayer(16,32,3,1,1),
#                                    *convlayer(32,64,3,2,1),
#                                    ResidualBlock(64),
#                                    ResidualBlock(64),
#                                    ResidualBlock(64),
#                                    ResidualBlock(64),
#                                    *convlayer(64,32,3,1,1),
#                                    nn.Conv2d(32,out_channels,3,1,1), 
#                                    self.out_activation)       

#     def forward(self, x):
#         return self.model(x)    

