# -*- coding: utf-8 -*-
import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np
from NL_gen import mac_64, mac_64_kernel, mac_64_multi_kernel
import pandas as pd
import os, sys, yaml
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from numpy import savetxt
from functionals.functionals import Conv2D
from models.simple_cnn import SimpleCNN



config = yaml.load(open('kt_qy_imc.yaml', 'r'), Loader=yaml.FullLoader)
print(config)


bias = torch.load('bias.pt')
stride = torch.load('stride.pt')
padding = torch.load('padding.pt')
groups = torch.load('groups.pt')
dilation = torch.load('dilation.pt')

I_NL = pd.read_excel('data_I-nonlinearity.xlsx').to_numpy()[:, 1]

I_NL = dict(enumerate(I_NL.flatten(), 0))


x = torch.load('x.pt',map_location='cuda:0').cpu().numpy()
weight =  torch.load('w.pt',map_location='cuda:0').cpu().numpy()
stride = stride[0]

# todo : acc_length is what analog sum length? then it should be 8 
acc_length = 64


batch = x.shape[0]
ch = x.shape[1]
feature_size = int(x.shape[2]/stride)
ks = weight.shape[2]
kernel_num = weight.shape[0]
round = int(np.ceil(ch/acc_length))
print(f"shape of x:{x.shape} shape of weight:{weight.shape}")
nl_err = 0
if ch < acc_length:
    ch = ch
else:
    ch = acc_length

if ks ==1:
    x = x
elif ks ==3:
    # x = F.pad(x, (1, 1, 1, 1), "constant", 0)
    x = np.pad(x, pad_width = ((0,0), (0,0), (1,1), (1,1)), mode = "constant", constant_values=0)

# vec_mac64 = np.vectorize((lambda x,y: mac_64(x, y, I_NL, NL_mode = False)), signature='(m,n),(m,n)->(m)')

#  *********************** single thread ***********************
output_features = np.zeros((batch, kernel_num, feature_size, feature_size))


model = SimpleCNN(config)
output_features1, layer_total_latency, layer_total_energy = model.forward((3, 32, 32, 3))

print(f"output feature size:{output_features1.shape}")

# output_features, layer_total_latency, layer_total_energy\
#   = conv2D(x, weight, bias, stride, padding, dilation, groups, config)

# model = SimpleCNN()

# print(f"***** number of total latency :{layer_total_latency} *****")
# savetxt('feat_debug.csv', output_features2[0,0], delimiter=',')
# print(output_features2[0,0])

# of_shape = output_features.shape
# identical = True
# print(f"shape is :{of_shape}")
# for i in range(of_shape[0]):
#     for j in range(of_shape[1]):
#         for k in range(of_shape[2]):
#             for q in range(of_shape[3]):
#                 if np.abs(output_features1[i,j,k,q] - output_features2[i,j,k,q]) > 0.00001:
                    # identical = False
                    # print(f"{i},{j},{k},{q} is different")
# print(f"the results are identical? {identical}")




# print(f"identical results:{False not in (output_features1 == output_features2)}")
# print(np.argwhere((output_features1 == output_features2)==False))

# output_features = 1
# error = np.array(output[0].cpu() -output_features[0])



