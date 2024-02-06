import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from NL_gen import mac_64, mac_64_kernel, mac_64_multi_kernel

def instantiate_x_weight_bias(x, weight, bias):
    # is no real input x, weight, bias given, then create a zero tensor
    if isinstance(x, tuple):
        x = np.zeros(x)
    
    if isinstance(weight, tuple):
        weight = np.zeros(weight)
    else:
        weight = weight.numpy()
    
    if isinstance(bias, tuple):
        bias = np.zeros(bias)
    else:
        bias = bias.numpy()
    
    return x, weight, bias


def non_linear_latency_perc(aa):
    # calculate the percentage latency according to KT's model
    if (aa == 0) or (aa == 1):
        return 1/7
    else:
        return aa/77

class Conv2D():

    I_NL = None

    def __init__(self, weight, bias, stride, padding, config):
        self.weight = weight
        self.bias = bias
        self.stride = stride[0]
        assert stride[0] == stride[1]
        self.padding = padding
        self.config = config
        self.acc_length = config['basic_paras']['num_row_per_chunk']
        self.allowed_max_analog_macs = config['basic_paras']['num_chunks'] * config['basic_paras']['num_row_per_chunk']
        self.I_NL = Conv2D.I_NL

    def forward(self, x):
        x, self.weight, self.bias = instantiate_x_weight_bias(x, self.weight, self.bias)
        print(f"self.weight shape is {self.weight.shape}")
        # shape related parameters
        batch, ch = x.shape[0], x.shape[1]
        feature_size = int(x.shape[2] / self.stride)
        kernel_num, ks = self.weight.shape[0], self.weight.shape[2]
        ch = ch if ch < self.acc_length else self.acc_length
        round = int(np.ceil(x.shape[1] / self.acc_length))
        # print(f"channel is {x.shape[1]} round is {round}")

        if x.shape[1] * ks * ks > self.allowed_max_analog_macs:
            raise Exception(f"kernel entries :{x.shape[1] * ks * ks} is larger than the number of analog values {self.allowed_max_analog_macs} in the analog array! Not supported yet")
            

        # calculate the number of bits of weight for this kernel
        bits_of_weight_this_kernel = x.shape[1] * ks * ks * self.config['basic_paras']['bits_per_weight']

        output_features = np.zeros((batch, kernel_num, feature_size, feature_size))
        if ks == 3:
            x = np.pad(x, pad_width=((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0)
            # x=F.pad(x, (1, 1, 1, 1), "constant", 0)

        layer_total_latency = 0
        layer_total_energy = 0
        layer_total_ops = 0

        for i in range(0, batch):
            for kn in tqdm(range(0, kernel_num)):
                
                # calculate the IO energy and latency
                weight_io_energy = self.config['energy']['per_bit_weight_io_energy'] \
                                    * bits_of_weight_this_kernel
                weight_io_latency = self.config['latency']['per_bit_weight_io_latency'] \
                                    * bits_of_weight_this_kernel
                layer_total_latency += weight_io_latency
                layer_total_energy += weight_io_energy

                for h in range(0, feature_size): #output feature size
                    for w in range(0, feature_size):
                        psum = 0
                        num_of_macs_this_kernel = 0

                        aa_max= 0
                        for kh in range(0, ks):
                            for kw in range(0, ks):
                                for s in range(0, round):
                                    # for c in range(0, ch):

                                    # if energy cost is modelled as dependent on the analog values
                                    if self.config['logging_control']['consider_analog_value_dependency']:
                                        # raise NotImplementedError
                                        print(f"i:{i}, kn:{kn}, h:{h}, w:{w}, kh:{kh}, kw:{kw}, s:{s}")
                                        aa = x[i, (ch * s): (ch * s + ch + 1), h * self.stride + kh, w * self.stride + kw]
                                        ww = self.weight[kn, (ch * s): (ch * s + ch), kh, kw]
                                        # get the maximum magnitude activation value
                                        aa_max= np.max(np.abs(aa), aa_max)
                                        if aa.size ==3:
                                            psum = psum + np.dot(aa, ww)
                                        else:
                                            psum = psum + mac_64(aa, ww, self.I_NL, NL_mode = False)
                                    
                                    num_of_macs_this_kernel += self.acc_length

                        # if num_of_macs_this_kernel > self.allowed_max_analog_macs:
                        #     raise Exception(f"kernel entries :{num_of_macs_this_kernel} is larger than the number of analog values {self.allowed_max_analog_macs} in the analog array! Not supported yet")
                        
                        # print(f"kernel entries :{num_of_macs_this_kernel} {self.allowed_max_analog_macs} in the analog array")
                        # calculate the total ops 
                        layer_total_ops += (num_of_macs_this_kernel*2)

                        if self.config['logging_control']['consider_analog_value_dependency']:
                            print(f"aa_max is {aa_max}, percent is {non_linear_latency_perc(aa_max)}")
                            non_linear_latency = non_linear_latency_perc(aa_max)*self.config['latency']['per_mac_latency']
                            layer_total_latency += non_linear_latency

                            layer_total_energy += 0 # ZW: TODO 
                        else:
                            layer_total_latency += self.config['latency']['per_bank_mac_latency']
                        
                            layer_total_energy += self.config['energy']['per_bank_mac_energy']

                        if self.config['logging_control']['consider_analog_value_dependency']:
                            output_features[i, kn, h, w] = psum

        output_features = torch.from_numpy(output_features).cuda()
        return output_features, layer_total_latency, layer_total_energy, layer_total_ops
    

class BatchNorm2D():
    def __init__(self, running_mean, running_var, gamma, beta, config):
        self.running_mean = running_mean
        self.running_var = running_var
        self.gamma = gamma
        self.beta = beta
        self.config = config

    def forward(self, x):
        assert x.shape[1] == self.running_mean.shape[0] # channel number should equal to the running mean channels
        assert x.shape[1] == self.running_var.shape[0]
        x = (x - self.running_mean)/(self.running_var + 1e-5).sqrt() * self.gamma + self.beta


class I_NL_ReLU():
    def __init__(self, I_NL, config):
        self.I_NL = I_NL
        self.config = config

    def forward(self, x):
        leq_zero = x <= 0
        laq_zero = x > 0
        x[laq_zero] = x[laq_zero] * self.I_NL
        x[leq_zero] = 0
        return x


class FC():
    def __init__(self, weight, bias, config):
        # FC not needed for E2E. but maybe for other task it's needed
        raise NotImplementedError
        # FC weights is described as y = Wx + b, if x is 10x1, W is 5x10, then y is 5x1
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        # assuming input to FC is already flattened, namely shape is (bs, flattened_dim)
        x, self.weight, self.bias = instantiate_x_weight_bias(x, self.weight, self.bias)  

        output_features = np.zeros((x.shape[0], self.weight.shape[0]))

        layer_total_latency = 0
        layer_total_energy = 0

        for i in range(0, x.shape[0]):
            output_features[i] = np.dot(self.weight, x[i]) + self.bias

        return output_features, 

