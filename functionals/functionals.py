import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from NL_gen import mac_64, mac_64_kernel, mac_64_multi_kernel


class Conv2D():
    def __init__(self, weight, bias, stride, padding, config):
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.config = config
        self.acc_length = config['basic_paras']['num_row_per_chunk']

    def forward(self, x):
        # is no real input x, weight, bias given, then create a zero tensor
        if isinstance(x, tuple):
            x = np.zeros(x)
        else:
            x = x.cpu().numpy()
        
        if isinstance(self.weight, tuple):
            self.weight = np.zeros(self.weight)
        else:
            self.weight = self.weight.cpu().numpy()
        
        if isinstance(self.bias, tuple):
            self.bias = np.zeros(self.bias)
        else:
            self.bias = self.bias.cpu().numpy()

        batch = x.shape[0]
        ch = x.shape[1]
        feature_size = int(x.shape[2] / self.stride[0])
        ks = self.weight.shape[2]
        kernel_num = self.weight.shape[0]

        ch = ch if ch < self.acc_length else self.acc_length
        self.round = int(np.ceil(ch / self.acc_length))

        if ks == 3:
            x = np.pad(x, pad_width=((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0)
            # x=F.pad(x, (1, 1, 1, 1), "constant", 0)

        layer_total_latency = 0
        layer_total_energy = 0

        for i in range(0, batch):
            for kn in tqdm(range(0, kernel_num)):
                
                # calculate the IO energy and latency
                bits_of_weight_this_kernel = ch * ks * ks * self.config['basic_paras']['bits_per_weight']
                weight_io_energy = self.config['energy']['per_bit_weight_io_energy'] \
                                    * bits_of_weight_this_kernel
                weight_io_latency = self.config['latency']['per_bit_weight_io_latency'] \
                                    * bits_of_weight_this_kernel
                layer_total_latency += weight_io_latency
                layer_total_energy += weight_io_energy

                for h in range(0, feature_size):
                    for w in range(0, feature_size):
                        psum = 0
                        num_of_macs_this_kernel = 0
                        for kh in range(0, ks):
                            for kw in range(0, ks):
                                for s in range(0, round):
                                    # for c in range(0, ch):

                                    # if energy cost is modelled as dependent on the analog values
                                    if self.config['logging_control']['consider_analog_value_dependency']:
                                        aa = x[i, (ch * s): (ch * s + ch), h * self.stride + kh, w * self.stride + kw]
                                        ww = self.weight[kn, (ch * s): (ch * s + ch), kh, kw]
                                        if aa.size ==3:
                                            psum = psum + np.dot(aa, ww)
                                        else:
                                            psum = psum + mac_64(aa, ww, I_NL, NL_mode = False)
                                    
                                    num_of_macs_this_kernel += self.acc_length

                        if num_of_macs_this_kernel > (self.config['basic_paras']['num_chunks'] * self.config['basic_paras']['num_row_per_chunk']):
                            raise Exception("kernel entries is larger than the number of analog values in the analog array! Not supported yet")
                        
                        layer_total_latency += self.config['latency']['per_bank_mac_latency']
                        layer_total_energy += self.config['energy']['per_bank_mac_energy']

                        output_features[i, kn, h, w] = psum
        output_features = torch.from_numpy(output_features).cuda()
        return output_features, layer_total_latency, layer_total_energy


# class FC():
#     def __init__(self, weight, bias, config):
#         self.weight = weight
#         self.bias = bias

#     def forward(self, x):



# def conv2D(x, weight, bias, stride, padding, dilation, groups, config):

#     # is no real input x, weight, bias given, then create a zero tensor
#     if isinstance(x, tuple):
#         x = np.zeros(x)
#     else:
#         x = x.cpu().numpy()
    
#     if isinstance(weight, tuple):
#         weight = np.zeros(weight)
#     else:
#         weight = weight.cpu().numpy()
    
#     if isinstance(bias, tuple):
#         bias = np.zeros(bias)
#     else:
#         bias = bias.cpu().numpy()
    
#     stride = stride[0]
#     batch = x.shape[0]
#     ch = x.shape[1]
#     feature_size = int(x.shape[2] / stride)
#     ks = weight.shape[2]
#     kernel_num = weight.shape[0]

#     acc_length = config['basic_paras']['num_row_per_chunk']
#     round = int(np.ceil(ch / acc_length))
#     output_features = np.zeros((batch, kernel_num, feature_size, feature_size))
#     if ch < acc_length:
#         ch = ch
#     else:
#         ch = acc_length

#     if ks == 3:
#         x = np.pad(x, pad_width=((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0)
#         # x=F.pad(x, (1, 1, 1, 1), "constant", 0)

#     layer_total_latency = 0
#     layer_total_energy = 0

#     for i in range(0, batch):
#         for kn in tqdm(range(0, kernel_num)):
            
#             # calculate the IO energy and latency
#             bits_of_weight_this_kernel = ch * ks * ks * config['basic_paras']['bits_per_weight']
#             weight_io_energy = config['energy']['per_bit_weight_io_energy'] \
#                                 * bits_of_weight_this_kernel
#             weight_io_latency = config['latency']['per_bit_weight_io_latency'] \
#                                 * bits_of_weight_this_kernel
#             layer_total_latency += weight_io_latency
#             layer_total_energy += weight_io_energy

#             for h in range(0, feature_size):
#                 for w in range(0, feature_size):
#                     psum = 0
#                     num_of_macs_this_kernel = 0
#                     for kh in range(0, ks):
#                         for kw in range(0, ks):
#                             for s in range(0, round):
#                                 # for c in range(0, ch):

#                                 # if energy cost is modelled as dependent on the analog values
#                                 if config['logging_control']['consider_analog_value_dependency']:
#                                     aa = x[i, (ch * s): (ch * s + ch), h * stride + kh, w * stride + kw]
#                                     ww = weight[kn, (ch * s): (ch * s + ch), kh, kw]
#                                     if aa.size ==3:
#                                         psum = psum + np.dot(aa, ww)
#                                     else:
#                                         psum = psum + mac_64(aa, ww, I_NL, NL_mode = False)
                                
#                                 num_of_macs_this_kernel += acc_length

#                     if num_of_macs_this_kernel > (config['basic_paras']['num_chunks'] * config['basic_paras']['num_row_per_chunk']):
#                          raise Exception("kernel entries is larger than the number of analog values in the analog array! Not supported yet")
                    
#                     layer_total_latency += config['latency']['per_bank_mac_latency']
#                     layer_total_energy += config['energy']['per_bank_mac_energy']

#                     output_features[i, kn, h, w] = psum
#     output_features = torch.from_numpy(output_features).cuda()
#     return output_features, layer_total_latency, layer_total_energy