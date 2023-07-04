# -*- coding: utf-8 -*-
import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np
from NL_gen import mac_64, mac_64_kernel, mac_64_multi_kernel
import pandas as pd
import os, sys
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from numpy import savetxt
x = torch.load('x.pt').cpu()
weight =  torch.load('w.pt').cpu()
bias = torch.load('bias.pt')
stride = torch.load('stride.pt')
padding = torch.load('padding.pt')
groups = torch.load('groups.pt')
dilation = torch.load('dilation.pt')
output = F.conv2d(x, weight,  bias,  stride,  padding, dilation,  groups)

I_NL = pd.read_excel('data_I-nonlinearity.xlsx').to_numpy()[:, 1]

I_NL = dict(enumerate(I_NL.flatten(), 0))

# print(I_NL[0])
x = torch.load('x.pt').cpu().numpy()
weight =  torch.load('w.pt').cpu().numpy()
stride = stride[0]
acc_length = 64


batch = x.shape[0]
ch = x.shape[1]
feature_size = int(x.shape[2]/stride)
ks = weight.shape[2]
kernel_num = weight.shape[0]
round = int(np.ceil(ch/acc_length))

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

#  ********************multiprocessing********************
def multiproc(i, h):
    output_feature_thread = np.zeros((kernel_num, feature_size))
    for w in range(0, feature_size):
        for kn in range(0, kernel_num):
            psum = 0
            for kh in range(0, ks):
                for kw in range(0, ks):
                    for s in range(0, round):
                        # for c in range(0, ch):
                        aa = x[i, (ch*s): (ch*s+ ch), h*stride+kh, w*stride+kw]
                        ww = weight[kn, (ch*s): (ch*s+ ch), kh, kw]
                        psum = psum + mac_64(aa, ww, I_NL, NL_mode = True)
                        psum = psum + nl_err
            output_feature_thread[kn, w] = psum
    return output_feature_thread

output_features = np.zeros((batch, kernel_num, feature_size, feature_size))
for i in tqdm(range(0, batch)):
    # for h in tqdm(range(0, feature_size)): parallelizing this for-loop
    with ProcessPoolExecutor(max_workers=int(os.cpu_count())) as executor:
        futures = executor.map(
            multiproc,
            [i for _ in range(feature_size)], range(0, feature_size)
        )
    # aggregate results
    futures = list(futures)
    # print(futures)
    for h in range(0, feature_size):
        output_features[i, :, h, :] = futures[h]

output_features1 = output_features
# savetxt('feat1.csv', output_features1[0,0], delimiter=',')
# print(output_features1[0,0])
#  ********************multiprocessing********************
# def multiproc_vec_kernel(i, h):
#     output_feature_thread = np.zeros((kernel_num, feature_size))
#     for w in range(0, feature_size):
#         for kn in range(0, kernel_num):
#             psum = np.zeros(ks*ks)
#             # for kh in range(0, ks):
#                 # for kw in range(0, ks):
#                     # for s in range(0, round):
#                         # for c in range(0, ch):
#             aa = x[i, :ch, h*stride:h*stride+ks, w*stride:w*stride+ks].reshape((ch, ks*ks))
#             ww = weight[kn, : ch, :ks, :ks].reshape((ch, ks*ks))
#             psum = psum + mac_64_kernel(aa, ww, I_NL, NL_mode = True, acc_length = acc_length)
#             psum = psum + nl_err
#             output_feature_thread[kn, w] = np.sum(psum)
#     return output_feature_thread

# output_features = np.zeros((batch, kernel_num, feature_size, feature_size))
# for i in tqdm(range(0, batch)):
#     # for h in tqdm(range(0, feature_size)): parallelizing this for-loop
#     # with ProcessPoolExecutor(max_workers=int(os.cpu_count())) as executor:
#     with ProcessPoolExecutor(max_workers=int(os.cpu_count())) as executor:
#         futures = executor.map(
#             multiproc_vec_kernel,
#             [i for _ in range(feature_size)], range(0, feature_size)
#         )
#     # aggregate results
#     futures = list(futures)
#     # print(futures)
#     for h in range(0, feature_size):
#         output_features[i, :, h, :] = futures[h]


def multiproc_vec_kernel(i, h):
    output_feature_thread = np.zeros((kernel_num, feature_size))
    for w in range(0, feature_size):
        # for kn in range(0, kernel_num):
        psum = np.zeros((kernel_num, ks*ks))
        # for kh in range(0, ks):
            # for kw in range(0, ks):
                # for s in range(0, round):
                    # for c in range(0, ch):
        aa = x[i, :ch, h*stride:h*stride+ks, w*stride:w*stride+ks].reshape((ch, ks*ks))
        ww = weight[:kernel_num, : ch, :ks, :ks].reshape((kernel_num, ch, ks*ks))
        psum = psum + mac_64_multi_kernel(aa, ww, I_NL, NL_mode = True, acc_length = acc_length)
        psum = psum + nl_err
        output_feature_thread[:, w] = np.sum(psum,axis=1)
    return output_feature_thread

output_features = np.zeros((batch, kernel_num, feature_size, feature_size))
for i in tqdm(range(0, batch)):
    # for h in tqdm(range(0, feature_size)): parallelizing this for-loop
    # with ProcessPoolExecutor(max_workers=int(os.cpu_count())) as executor:
    with ProcessPoolExecutor(max_workers=int(os.cpu_count())) as executor:
        futures = executor.map(
            multiproc_vec_kernel,
            [i for _ in range(feature_size)], range(0, feature_size)
        )
    # aggregate results
    futures = list(futures)
    # print(futures)
    for h in range(0, feature_size):
        output_features[i, :, h, :] = futures[h]

output_features2 = output_features
savetxt('feat2.csv', output_features2[0,0], delimiter=',')
print(output_features2[0,0])

of_shape = output_features.shape
identical = True
print(f"shape is :{of_shape}")
for i in range(of_shape[0]):
    for j in range(of_shape[1]):
        for k in range(of_shape[2]):
            for q in range(of_shape[3]):
                if np.abs(output_features1[i,j,k,q] - output_features2[i,j,k,q]) > 0.00001:
                    identical = False
                    # print(f"{i},{j},{k},{q} is different")
print(f"the results are identical? {identical}")


#  single thread
# output_features = np.zeros((batch, kernel_num, feature_size, feature_size))
# for i in tqdm(range(0, batch)):
#     for kn in range(0, kernel_num):
#         for h in range(0, feature_size):
#             for w in range(0, feature_size):
#                 psum = 0
#                 for kh in range(0, ks):
#                     for kw in range(0, ks):
#                         for s in range(0, round):
#                             # for c in range(0, ch):
#                             aa = x[i, (ch*s): (ch*s+ ch), h*stride+kh, w*stride+kw]
#                             ww = weight[kn, (ch*s): (ch*s+ ch), kh, kw]
#                             psum = psum + mac_64(aa, ww, I_NL, NL_mode = True)
#                             psum = psum + nl_err
#                 output_features[i, kn, h, w] = psum
# output_features2 = output_features



# print(f"identical results:{False not in (output_features1 == output_features2)}")
# print(np.argwhere((output_features1 == output_features2)==False))

# output_features = 1
error = np.array(output[0].cpu() -output_features[0])



def conv_sim(x, weight, bias, stride, padding, dilation, groups, acc_length):
    I_NL = pd.read_excel('data_I-nonlinearity_64v2.xlsx').to_numpy()[:, 1]
    x=x.cpu().numpy()
    weight=weight.cpu().numpy()
    stride = stride[0]
    batch = x.shape[0]
    ch = x.shape[1]
    feature_size = int(x.shape[2] / stride)
    ks = weight.shape[2]
    kernel_num = weight.shape[0]
    round = int(np.ceil(ch / acc_length))
    output_features = np.zeros((batch, kernel_num, feature_size, feature_size))
    if ch < acc_length:
        ch = ch
    else:
        ch = acc_length

    if ks == 3:
        x = np.pad(x, pad_width=((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0)
        # x=F.pad(x, (1, 1, 1, 1), "constant", 0)

    for i in range(0, batch):
        for kn in tqdm(range(0, kernel_num)):
            for h in range(0, feature_size):
                for w in range(0, feature_size):
                    psum = 0
                    for kh in range(0, ks):
                        for kw in range(0, ks):
                            for s in range(0, round):
                                # for c in range(0, ch):
                                aa = x[i, (ch * s): (ch * s + ch), h * stride + kh, w * stride + kw]
                                ww = weight[kn, (ch * s): (ch * s + ch), kh, kw]
                                if aa.size ==3:
                                    psum = psum + np.dot(aa, ww)
                                else:
                                    psum = psum + mac_64(aa, ww, I_NL, NL_mode = False)
                    output_features[i, kn, h, w] = psum
    output_features = torch.from_numpy(output_features).cuda()
    return output_features