from collections import OrderedDict
from icecream import ic 
import os
import numpy as np
import math
import torch 
import torch.nn as nn
# from bn_fold import fuse_bn_recursively
import random
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import time
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import Dataset, DataLoader 
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
import utils 
import NVP_v1_model, training_nvp, PTQ
import local_datasets
import argparse
import pandas as pd


seed = np.random.randint(1000)
#50
np.random.seed(seed)
torch.manual_seed(seed)


########### 
# Accelerator configuration
# !! make sure they match the accelerator RTL code
########### 
activation_precision            = 8
ACTIVATION_MIN_VALUE            = -1*2**(activation_precision-1)
ACTIVATION_MAX_VALUE            = 2**(activation_precision-1)-1
weight_precision                = 8
bias_precision                  = 32
bias_precision_axi              = 32
quantization_scale_precision    = 18
axi_word_width                  = 64
weight_axi_word_width           = axi_word_width
NUMBER_OF_PES_PER_ARRAY         = 16


def load_torchscript_model(model_filepath, device):
    model = torch.jit.load(model_filepath, map_location=device)
    return model

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model_name", type=str, default="demo_model",
                help="model name")
ap.add_argument("-dir", "--savedir", type=str, default="nvp_NN/",
                help="directory for saving the model parameters and training statistics")
ap.add_argument("-s", "--seed", type=int, default=0,
                help="seed for random initialization")
ap.add_argument("-e", "--n_epochs", type=int, default=80,
                help="number of training epochs")   
ap.add_argument("-l", "--log_interval", type=int, default=10,
                help="number of batches after which to evaluate model (and logged)")   
ap.add_argument("-crit", "--convergence_crit", type=int, default=50,
                help="stop-criterion for convergence: number of evaluations after which model is not improved")   
ap.add_argument("-bin", "--binary_stimulation", type=bool, default=True,
                help="use quantized (binary) instead of continuous stimulation protocol")   
ap.add_argument("-sim", "--simulation_type", type=str, default="regular",
                help="'regular' or 'personalized' phosphene mapping") 
ap.add_argument("-in", "--input_channels", type=int, default=1,
                help="only grayscale (single channel) images are supported for now")   
ap.add_argument("-out", "--reconstruction_channels", type=int, default=1,
                help="only grayscale (single channel) images are supported for now")     
ap.add_argument("-act", "--out_activation", type=str, default="sigmoid",
                help="use 'sigmoid' for grayscale reconstructions, 'softmax' for boundary segmentation task")   
ap.add_argument("-d", "--dataset", type=str, default="characters",
                help="'charaters' dataset and 'ADE20K' are supported")   
ap.add_argument("-dev", "--device", type=str, default="cpu",
                help="e.g. use 'cpu' or 'cuda:0' ")   
ap.add_argument("-n", "--batch_size", type=int, default=30,
                help="'charaters' dataset and 'ADE20K' are supported")   
ap.add_argument("-opt", "--optimizer", type=str, default="adam",
                help="only 'adam' is supporte for now")   
ap.add_argument("-lr", "--learning_rate", type=float, default=0.0001,
                help="Use higher learning rates for VGG-loss (perceptual reconstruction task)")  
ap.add_argument("-rl", "--reconstruction_loss", type=str, default='mse',
                help="'mse', 'VGG' or 'boundary' loss are supported ") 
ap.add_argument("-p", "--reconstruction_loss_param", type=float, default=0,
                help="In perceptual condition: the VGG layer depth, boundary segmentation: cross-entropy class weight") 
ap.add_argument("-L", "--sparsity_loss", type=str, default='L1',
                help="choose L1 or L2 type of sparsity loss (MSE or L1('taxidrivers') norm)") 
ap.add_argument("-k", "--kappa", type=float, default=0.01,
                help="sparsity weight parameter kappa")    
# ap.add_argument("-nvp", "--NVP_v1", action='store_true', default=False)
# ap.add_argument("-nvp_qat", "--NVP_v1_QAT", action='store_true', default=False)
cfg = pd.Series(vars(ap.parse_args()))

print(cfg)


cuda_device = torch.device("cuda:0")
cpu_device = torch.device("cpu:0")

model_dir = cfg.savedir
model_filename = cfg.model_name+'_best_encoder.pth'
quantized_model_filename = cfg.model_name+'_best_encoder_quantized.pth'
quantized_jit_model_filename = cfg.model_name+'_jit_best_encoder_quantized.pth'
model_filepath = os.path.join(model_dir, model_filename)
quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)
quantized_jit_model_filepath = os.path.join(model_dir, quantized_jit_model_filename)

""" loads the saved model parametes for given configuration <cfg> and returns the performance
metrics on the validation dataset. The <visualize> argument can be set equal to any positive
integer that represents the amount of example figures to plot."""
# Load configurations
models, dataset, optimization, train_settings = training_nvp.initialize_components(cfg)
encoder = models['encoder']
decoder = models['decoder']
simulator = models['simulator']
trainloader = dataset['trainloader']
valloader = dataset['valloader']
lossfunc = optimization['lossfunc']

# Load model parameters
encoder.load_state_dict(torch.load(os.path.join(cfg.savedir,cfg.model_name+'_best_encoder.pth')))
decoder.load_state_dict(torch.load(os.path.join(cfg.savedir,cfg.model_name+'_best_decoder.pth')))
encoder.eval()
decoder.eval()

# create quantized encoder
fused_encoder = copy.deepcopy(encoder)
for module_name, module in fused_encoder.named_children():
    for basic_block_name, basic_block in module.named_children():
        if(isinstance(basic_block, NVP_v1_model.convBlock)):
            torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"]], inplace=True)  
        if(isinstance(basic_block, NVP_v1_model.ResidualBlock)):
            torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"]], inplace=True)  
quantized_encoder = NVP_v1_model.Quantized_E2E_Encoder(model_fp32=fused_encoder)
quantization_config = torch.quantization.QConfig(
activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine), 
weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric))
quantized_encoder.qconfig = quantization_config
torch.quantization.prepare(quantized_encoder, inplace=True)
quantized_encoder = torch.quantization.convert(quantized_encoder, inplace=True)
quantized_encoder.eval()

test_input = torch.rand(size=(1,1,4,4)).to(cpu_device)
model = quantized_encoder.eval()

# Load quantized encoder.
quantized_encoder.load_state_dict(torch.load(os.path.join(cfg.savedir,cfg.model_name+'_best_encoder_quantized.pth')))
quantized_jit_encoder = load_torchscript_model(model_filepath=quantized_jit_model_filepath, device=cpu_device)

## check the loaded encoders
# int8_metrics = PTQ.evaluate_model(cfg, quantized_encoder, decoder, simulator, valloader, device=cpu_device, quantized=True, criterion=None, visualize=5, savefig=False)
# int8_metrics = PTQ.evaluate_model(cfg, quantized_jit_encoder, decoder, simulator, valloader, device=cpu_device, quantized=True, criterion=None, visualize=5, savefig=False)
# exit()


########### 
# create output compiled NN directory 
########### 
# out_directory_name = "phosphene_NN_no_skips_" + str(NUMBER_OF_PES_PER_ARRAY) +"_pes_"+ str(activation_precision) +"_bits_"+ str(axi_word_width) + "_axi"
out_directory_name = "compiled_NN"
ic(out_directory_name)
if not os.path.exists(out_directory_name):
    os.mkdir(out_directory_name)
    print("Directory " , out_directory_name ,  " Created ")
else:    
    print("Directory " , out_directory_name ,  " already exists")

# ########### 
# # weight quantization 
# # !! must match the accelerator
# ###########  
qweight_min = -128.0 #-128.0
qweight_max = 127.0 #127.0
# calibration_qweight_min = -1
# calibration_qweight_max = 1
# qweight_scale = round((qweight_max-qweight_min)/(calibration_qweight_max-calibration_qweight_min))
# # qweight_scale = round(128/calibration_qweight_max)

# ########### 
# # activation quantization 
# # !! must match the accelerator
# ########### 
qactivation_min = 0
qactivation_max = 255
q_input_activation_scale = quantized_encoder.quant.scale.item()
q_output_activation_scale = q_input_activation_scale
# ic(q_input_activation_scale) 

# ########### 
# # bias quantization 
# # !! must match the accelerator
# ########### 
qbias_min = -(2**(bias_precision-1))
qbias_max = (2**(bias_precision-1))-1
# qbias_scale = qweight_scale*q_input_activation_scale

quantization_scale_list = []

########### 
# create random input activations 
########### 
ROWS       = 128
COLS       = 128
CHANNELS   = 1
i_rows     = ROWS
i_cols     = COLS
i_channels = CHANNELS
i_sparsity = 0.00 # change input sparsity. 


# Forward pass (validation set)
image,label = next(iter(valloader))
image.to(cfg.device)
image = image[0]
label = label[0]
image = image[None, :]
# quantized_input_mat = encoder.quant(image)
# conv_output = quantized_input_mat

# image = torch.rand(size=(1, i_channels, i_rows, i_cols)).to(cpu_device)
# quantized_input_mat = model.quant(image)
quantized_input_mat = model.quant(image)
quantized_input_mat_int = quantized_input_mat.int_repr()
conv_output = quantized_input_mat
real_conv_output = image

ic(image)
ic(quantized_input_mat_int)

########### 
# Compression function 
# The input channels are divided into sub-tiles. The number of sub-tiles is equal to number_of_channels/activation_precision. 
# For example, when activation precision is 8, this means that each SM segment maps 8 activation pixels. 
# Here, the SM size is fixed to 8. 
# Note that if the number of channels is less than 8 (e.g. first layer), the num_sub_tiles is set to 1.
########### 
def sm_compression(input_mat, n_rows, n_cols, num_sub_tiles, channels_per_sub_tile, shift_dont_clip=False):
    compressed_mats = []
    for row in range(n_rows):
        compressed_mats.append([])
        # ic(row)
        for col in range(n_cols):
            for subtile in range(num_sub_tiles):
                # SM = "00000000"
                SM = list('00000000')
                for ch in range(channels_per_sub_tile):
                    if (input_mat[0, subtile*channels_per_sub_tile+ch, row, col] != 0):
                        SM[ch] = "1"
                compressed_mats[row].append(int(''.join(SM), 2))
                # if(shift_dont_clip):
                #         ic(SM)
                for ch in range(channels_per_sub_tile):
                    val = input_mat[0, subtile*channels_per_sub_tile+ch, row, col].item()
                    # if(shift_dont_clip):
                    #     ic(val)
                    if (val != 0.0):
                        if(shift_dont_clip):
                            # if(val<0.0):
                            #     val = val + 256
                            if (val<-128):
                                # ic(val)
                                val = -128
                            if (val>127):
                                # ic(val)
                                val = 127
                            if(val<0.0):
                                val = val + 256
                        else:
                            if (val<qactivation_min):
                                # ic(val)
                                val = qactivation_min
                            if (val>qactivation_max):
                                # ic(val)
                                val = qactivation_max
                        
                        compressed_mats[row].append(val)
        # if(shift_dont_clip):
        #     ic("---------------")
    return compressed_mats

########### 
# Save compressed matrices function
# Saves the compressed matrix in AXI words. 
# Only tested axi-width of 64
########### 
def save_compressed_mats(compressed_mats, saved_file_keyword, axi_word_width, activation_precision):
    row = 0
    for compressed_mat in compressed_mats:
        row = row + 1
        mod_8 = 8-len(compressed_mat)%8
        if(mod_8!=8): ## extend to multiple of 8 -> to fit buffer words with 64 bits
            for i in range(mod_8):
                compressed_mat.append(0)

        ### save as hex
        ### create buffer words (64 bits)
        compressed_mat_input_buffer = []
        number_of_words_per_axis_word = int(axi_word_width/activation_precision)
        # ic(len(compressed_mat))
        for i in range(0, len(compressed_mat), number_of_words_per_axis_word): # append into axi-bit words
            # ic(i)
            buffer_word = ""
            if(i+number_of_words_per_axis_word>len(compressed_mat)):
                remaining_words = len(compressed_mat) - i
                for j in range(0, remaining_words):
                    buffer_word +=  format(compressed_mat[i+j], '02x') 
                    # ic(compressed_mat[i+j])
                for j in range(remaining_words, number_of_words_per_axis_word):
                    buffer_word +=  format(0, '02x') 
            else:
                for j in range(0, number_of_words_per_axis_word):
                    # ic(compressed_mat[i+j])
                    buffer_word +=  format(compressed_mat[i+j], '02x') 
                    # if(compressed_mat[i+j]==136):
                    #     ic(compressed_mat[i+j])
            compressed_mat_input_buffer.append(buffer_word)
        # np.savetxt(out_directory_name+'/{}_{}.txt'.format(saved_file_keyword, row-1), compressed_mat_input_buffer, fmt='%s')
        np.savetxt('{}_{}.txt'.format(saved_file_keyword, row-1), compressed_mat_input_buffer, fmt='%s')

# #############
# # save input activations non-zero values and their coordinates
# #############
# for row in range(i_rows):
#     with open(out_directory_name+'/compressed_mat_{}_out.txt'.format(row), 'w') as f:
#         for col in range(i_cols):
#             f.write("-> col = {}\n".format(col))
#             for ch in range(i_channels):
#                 if (quantized_input_mat[0, ch, row, col] !=0):
#                     f.write("decoded word = {}, channel = {}\n".format(quantized_input_mat[0, ch, row, col], ch))
#                     # f.write('\n')
#             f.write("-----\n")
#             # f.write('\n')


#############
# Compress and save input activations
#############
num_input_channel_sub_tiles = max(i_channels//activation_precision, 1)
if(i_channels<activation_precision):
    channels_per_sub_tile = i_channels
else:
    channels_per_sub_tile = activation_precision
compressed_mats = sm_compression(quantized_input_mat_int, i_rows, i_cols, num_input_channel_sub_tiles, channels_per_sub_tile)
input_activation_dir = out_directory_name+"/input_activations"
if not os.path.exists(input_activation_dir):
    os.mkdir(input_activation_dir)
    print("Directory " , input_activation_dir ,  " Created ")
else:    
    print("Directory " , input_activation_dir ,  " already exists")
save_compressed_mats(compressed_mats, input_activation_dir+"/input_activation", axi_word_width, activation_precision)



#############
# Compilation loop
# Iterate over the layers one by one. First it runs the layer to generate the output activations. Then exports weights in axi words. 
# Finally it compresses and exports output activations to validate systemverilog simulation. 
#############
export_readable_format = False # if True, this exports the coordinates of the non-zero activations to debug the simulation.  
index = 0
bias_ = []
# for name, module in model.model_fp32.model.named_modules():
#     if(isinstance(module,torch.nn.intrinsic.quantized.ConvReLU2d)):
#         ic(index, name, module)
#         # ic(module.weight().int_repr())
#         # ic(module.bias())
#         ic(module.weight().q_scale())
#         ic(module.weight().q_zero_point())
#         ic(module.scale)
#         ic(module.zero_point)
# exit()


run_quantized_model = False
run_hw_simulation_model = not run_quantized_model
for name, module in model.named_modules():
    ic(index, name, module)
    ic(type(module)) 
    if(isinstance(module,torch.nn.quantized.Conv2d)):
        ic("bbbbbbbbbbbbbbbbbbbb")
    if(isinstance(module,torch.nn.intrinsic.quantized.ConvReLU2d) or isinstance(module,torch.nn.quantized.Conv2d)):
        # print(type(module))
        # print(module.weight.shape)
        # print(index, name, module)
        # print(index, name)
        # print(module.bias)

        # ic(index, name, module)
        ic("----------------------")

        weights = module.weight().int_repr()
        bias = module.bias()

        q_input_activation_scale = q_output_activation_scale
        qweight_scale = module.weight().q_scale()
        q_output_activation_scale = module.scale
        qbias_scale = qweight_scale*q_input_activation_scale

        ic(1/q_input_activation_scale, q_input_activation_scale)
        ic(1/q_output_activation_scale, q_output_activation_scale)
        ic(1/qweight_scale, qweight_scale)
        ic(1/qbias_scale, qbias_scale)

        i_rows     = conv_output.shape[2]
        i_cols     = conv_output.shape[3]
        i_channels = conv_output.shape[1]
        ic(weights.shape)
        n_kernels  = weights.shape[0]

        if(export_readable_format):
            #############
            # save activations non-zero values and their coordinates
            #############
            for row in range(i_rows):
                with open(out_directory_name+'/layer_{}_compressed_mat_{}_out.txt'.format(index,row), 'w') as f:
                    for col in range(i_cols):
                        f.write("-> col = {}\n".format(col))
                        for ch in range(i_channels):
                            if (conv_output[0, ch, row, col] !=0):
                                f.write("decoded word = {}, channel = {}\n".format(conv_output[0, ch, row, col], ch))
                                # f.write('\n')
                        f.write("-----\n")
                        # f.write('\n')

        # layer_i_directory = out_directory_name + "/layer_"+str(index) +"_"+ str(name)
        layer_i_directory = out_directory_name + "/layer_"+str(index)
        
        # ic(layer_i_directory)
        if not os.path.exists(layer_i_directory):
            os.mkdir(layer_i_directory)
            print("Directory " , layer_i_directory ,  " Created ")
        else:    
            print("Directory " , layer_i_directory ,  " already exists")

        with torch.no_grad(): 
            
            if(run_quantized_model):
                ## Run convolution and quantize output activations
                conv_output_tmp = module(conv_output)
            
            
            scale = (qweight_scale*q_input_activation_scale) / q_output_activation_scale
            ic(scale)
            quantization_scale_list.append(round((scale)*(2**quantization_scale_precision)))
            ic(quantization_scale_list[index])
            index += 1

            bias   = torch.nn.Parameter(torch.clip(torch.floor(bias/qbias_scale).float(),  min=qbias_min, max=qbias_max)) 
            
            if(run_quantized_model):
                ##### quantized model
                conv_output = conv_output_tmp
                conv_output_np = conv_output.int_repr().numpy()




            # if(index==1):
            #     simulated_conv_output = torch.nn.functional.conv2d(input=conv_output.int_repr().float(), weight=weights.float(), bias=bias, stride=module.stride[0], padding=module.padding) 
            # else:
            #     simulated_conv_output = torch.nn.functional.conv2d(input=conv_output.float(), weight=weights.float(), bias=bias, stride=module.stride[0], padding=module.padding) 
            # if(isinstance(module,torch.nn.intrinsic.quantized.ConvReLU2d)):
            #     simulated_conv_output = torch.nn.functional.relu(simulated_conv_output)
            #     ic("inside relu")
            #     # simulated_conv_output = (simulated_conv_output*quantization_scale_list[index-1])  // (2**quantization_scale_precision)
            #     simulated_conv_output = torch.floor((simulated_conv_output*quantization_scale_list[index-1])  / (2**quantization_scale_precision))
            # else:
            #     ic(simulated_conv_output.shape)
            #     # ic(simulated_conv_output.detach().numpy())
            #     ic(simulated_conv_output.detach().numpy()[0][0][0])
            #     # ic(simulated_conv_output.detach().numpy()[0][0][15])  
            #     simulated_conv_output = (simulated_conv_output*quantization_scale_list[index-1])  / (2**quantization_scale_precision)
            #     simulated_conv_output = torch.floor(simulated_conv_output)
            # # simulated_conv_output = (simulated_conv_output*quantization_scale_list[index-1])  // (2**quantization_scale_precision)
            # conv_output = simulated_conv_output
            # conv_output_np = conv_output.detach().numpy()



            if(run_hw_simulation_model):
                if(index==1):
                    simulated_conv_output = torch.nn.functional.conv2d(input=conv_output.int_repr().float(), weight=weights.float(), bias=bias, stride=module.stride[0], padding=module.padding) 
                else:
                    simulated_conv_output = torch.nn.functional.conv2d(input=conv_output.float(), weight=weights.float(), bias=bias, stride=module.stride[0], padding=module.padding) 
                
                if(isinstance(module,torch.nn.intrinsic.quantized.ConvReLU2d)):
                    simulated_conv_output = torch.nn.functional.relu(simulated_conv_output)
                    simulated_conv_output = (simulated_conv_output*quantization_scale_list[index-1]) // (2**quantization_scale_precision)
                else:
                    # ic(simulated_conv_output)
                    ic(simulated_conv_output.detach().numpy()[0][0][0])
                    # ic(simulated_conv_output.detach().numpy()[0][0][15])  
                    simulated_conv_output = (simulated_conv_output*quantization_scale_list[index-1])  / (2**quantization_scale_precision)
                    simulated_conv_output = torch.floor(simulated_conv_output)
                    ic(simulated_conv_output.detach().numpy()[0][0][0])
                    

                ##### hw-simulation model
                conv_output = simulated_conv_output
                conv_output_np = conv_output.detach().numpy()
            if(not isinstance(module,torch.nn.intrinsic.quantized.ConvReLU2d)):
                import sys
                np.set_printoptions(threshold=sys.maxsize)
                if(run_quantized_model):
                    # ic(module.zero_point)
                    conv_output_np = (conv_output_np-module.zero_point).view(dtype=np.int8)
                    ic(conv_output_np)
                else:    
                    ic(conv_output_np)
                

            # np.savetxt(layer_i_directory+'/relu_conv_output.txt', conv_output_np.flatten(order='C'), fmt='%s')
            np.savetxt(layer_i_directory+'/relu_conv_output.txt', conv_output_np[:,:,0,:].flatten(order='F'), fmt='%s')
            # n, bins, patches = plt.hist(conv_output_np.flatten(), 50, density=False, facecolor='g', alpha=0.75)
            # plt.show()
            # ic(conv_output.max())
            # ic(conv_output.min())

            # ic(conv_output_np[0][0])
        

        #############
        # save weights for axi transfer
        #############
        # ic(module.weight.shape)
        # numpy_weights = np.floor(module.weight.detach().numpy()).astype(int)
        numpy_weights = weights.numpy()
        # ic(numpy_weights.shape)
        # ic(numpy_weights.min())
        # ic(numpy_weights.max())
        weight_mats = [[],[],[]]
        word = []
        number_of_words_per_axis_word = int(weight_axi_word_width/weight_precision)
        number_of_weights_per_axis_word = min(number_of_words_per_axis_word, n_kernels)
        max_kernels_and_pes = max(NUMBER_OF_PES_PER_ARRAY, n_kernels) # to handle the case where n_kernels is less than NUMBER_OF_PES_PER_ARRAY
        number_of_steps = np.ceil(max_kernels_and_pes/number_of_words_per_axis_word).astype(int)
        for col in range(3): 
            weight_col = numpy_weights[:, :, :, col]
            # print("col= ", col)
            for row in range(3):
                for channel in range(max(i_channels,8)):
                    for step in range(number_of_steps):
                        word = []
                        for kernel_i in range(step*number_of_weights_per_axis_word, (step+1)*number_of_weights_per_axis_word):
                            if(kernel_i>=n_kernels or channel>=i_channels):
                                val = 0
                            else:    
                                val = weight_col[kernel_i, channel, row]
                            # if(val==128):
                            #     print("val= ", val)
                            if(val<0):
                                # print("val= ", val)
                                val = val + 256
                            word.append(val)
                        weight_mats[2-col].append(word)
        for col in range(3): # save words
            weight_array_i = []
            for word_i in range(len(weight_mats[col])): 
                # ic(i)
                buffer_word = ""
                if(number_of_words_per_axis_word>len(weight_mats[col][word_i])):
                    zero_remaining_words = number_of_words_per_axis_word - len(weight_mats[col][word_i]) 
                    # ic(zero_remaining_words)
                    for j in range(0, number_of_words_per_axis_word-zero_remaining_words):
                        buffer_word +=  format(weight_mats[col][word_i][j], '02x') 
                    for j in range(0, zero_remaining_words):
                        buffer_word +=  format(0, '02x') 
                else:
                    for j in range(0, number_of_words_per_axis_word):
                        buffer_word +=  format(weight_mats[col][word_i][j], '02x') 
                weight_array_i.append(buffer_word)
            np.savetxt(layer_i_directory+'/weight_array_{}.txt'.format(col), weight_array_i, fmt='%s')


        # bias   = torch.nn.Parameter(torch.clip(torch.floor(bias/qbias_scale).float(),  min=qbias_min, max=qbias_max)) 
        numpy_bias = np.floor(bias.detach().numpy()).astype(int)
        # if(module.bias != None):
        # if(bias == True):
        # numpy_bias = np.floor(module.bias.detach().numpy()).astype(int)
        print(numpy_bias)
        if(len(numpy_bias)==1):
            tmp_list = []
            tmp_list.append(numpy_bias[0])
            for i in range(activation_precision-1):
                tmp_list.append(0)
            numpy_bias = np.array(tmp_list)
            print(numpy_bias)
        bias_mat = []
        word = []
        number_of_bias_values_per_axis_word = int(weight_axi_word_width/bias_precision_axi)
        # number_of_bias_values_per_axis_word = min(number_of_words_per_axis_word, n_kernels)
        max_kernels_and_pes = max(NUMBER_OF_PES_PER_ARRAY, n_kernels) # to handle the case where n_kernels is less than NUMBER_OF_PES_PER_ARRAY
        number_of_steps = np.ceil(max_kernels_and_pes/number_of_bias_values_per_axis_word).astype(int)
        for bias_i in range(0, len(numpy_bias),number_of_bias_values_per_axis_word): 
            word = []
            if(len(numpy_bias) < number_of_bias_values_per_axis_word):
                for i in range(len(numpy_bias)):
                    val = numpy_bias[bias_i+i]
                    # print(val)
                    if(val<0):
                        # print("val= ", val)
                        val = val + (2**bias_precision)
                    word.append(val)
                for i in range(len(numpy_bias), number_of_bias_values_per_axis_word):
                    val = 0
                    word.append(val)
            else: 
                for i in range(number_of_bias_values_per_axis_word):
                    val = numpy_bias[bias_i+i]
                    # print(val)
                    if(val<0):
                        # print("val= ", val)
                        val = val + (2**bias_precision)
                    word.append(val)
            bias_mat.append(word)
        bias_array = []
        for word_i in range(len(bias_mat)): 
            # ic(i)
            buffer_word = ""
            for j in range(0, number_of_bias_values_per_axis_word):
            # ic(number_of_bias_values_per_axis_word)
            # for j in range(number_of_bias_values_per_axis_word-1, -1, -1):
                # ic(j)
                buffer_word +=  format(bias_mat[word_i][j], '08x') 
            bias_array.append(buffer_word)
        np.savetxt(layer_i_directory+'/bias_array.txt', bias_array, fmt='%s')

        

        #############
        # Compress and save output activations
        #############
        if(run_hw_simulation_model):
            conv_output_np = np.round(conv_output.numpy()).astype(int)
        o_channels      = conv_output_np.shape[1]
        o_rows          = conv_output_np.shape[2]
        o_cols          = conv_output_np.shape[3]
        n_output_pixels = o_rows * o_cols * o_channels
        # ic(conv_output_np)
        num_output_channel_sub_tiles = max(o_channels//activation_precision, 1)
        ic(index)
        if(o_channels<activation_precision):
            channels_per_sub_tile = o_channels
        else:
            channels_per_sub_tile = activation_precision
        if(index-1 == 13):
            shift_dont_clip = True
        else:
            shift_dont_clip = False
        output_activation_compressed_mats = sm_compression(conv_output_np, o_rows, o_cols, num_output_channel_sub_tiles, channels_per_sub_tile, shift_dont_clip=shift_dont_clip)
        save_compressed_mats(output_activation_compressed_mats, layer_i_directory+'/compressed_output_activation', axi_word_width, activation_precision)
        # if(index-1 != 13):
        #     output_activation_compressed_mats = sm_compression(conv_output_np, o_rows, o_cols, num_output_channel_sub_tiles, activation_precision)
        #     save_compressed_mats(output_activation_compressed_mats, layer_i_directory+'/compressed_output_activation', axi_word_width, activation_precision)
        # else:
        #     # ic(conv_output_np.shape)
        #     # ic(conv_output_np.flatten().shape)
        #     # ic(conv_output_np.flatten())
        #     # ic(max(conv_output_np.flatten()))
        #     # ic(min(conv_output_np.flatten()))
        #     mat_input_buffer = []
        #     # for val in conv_output_np.flatten():
        #     number_of_words_per_axis_word = int(axi_word_width/activation_precision)
            
        #     for i in range(0, conv_output_np.flatten().shape[0], number_of_words_per_axis_word): # append into axi-bit words
        #         # ic(i)
        #         buffer_word = ""
        #         if(i+number_of_words_per_axis_word>conv_output_np.flatten().shape[0]):
        #             remaining_words = conv_output_np.flatten().shape[0] - i
        #             for j in range(0, remaining_words):
        #                 val = conv_output_np.flatten()[i+j]
        #                 if(val<0.0):
        #                     val = val + 256
        #                 buffer_word +=  format(val, '02x') 
        #                 # ic(compressed_mat[i+j])
        #             for j in range(remaining_words, number_of_words_per_axis_word):
        #                 buffer_word +=  format(0, '02x') 
        #         else:
        #             for j in range(0, number_of_words_per_axis_word):
        #                 val = conv_output_np.flatten()[i+j]
        #                 if(val<0.0):
        #                     val = val + 256
        #                 buffer_word +=  format(val, '02x') 
        #                 # if(compressed_mat[i+j]==136):
        #                 #     ic(compressed_mat[i+j])
        #         mat_input_buffer.append(buffer_word)
        #     # np.savetxt(out_directory_name+'/{}_{}.txt'.format(saved_file_keyword, row-1), compressed_mat_input_buffer, fmt='%s')
        #     np.savetxt('{}.txt'.format(layer_i_directory+"/Final_layer_output_activations"), mat_input_buffer, fmt='%s')


        ## print sparsity
        conv_output_sparsity = np.zeros_like(conv_output_np)
        conv_output_sparsity[conv_output_np>0] = 1
        NNZ = (conv_output_sparsity).sum()
        output_activation_sparsity = (1-NNZ/n_output_pixels)*100
        ic(output_activation_sparsity)


#############
# Generate systemverilog test_NN_package 
# 
#############
# !! These parameters must match the accelerator RTL code. 
ACTIVATION_LINE_BUFFER_DEPTH              = 512
ACTIVATION_BANK_BIT_WIDTH                 = axi_word_width 
NUMBER_OF_READ_STREAMS                    = 3
# NUMBER_OF_ACTIVATION_LINE_BUFFERS         = NUMBER_OF_READ_STREAMS*2 # ultrascale
NUMBER_OF_ACTIVATION_LINE_BUFFERS         = NUMBER_OF_READ_STREAMS*2 # minized
ACTIVATION_BUFFER_BANK_COUNT              = (NUMBER_OF_PES_PER_ARRAY*activation_precision)//ACTIVATION_BANK_BIT_WIDTH
ACTIVATION_LINE_BUFFER_SIZE               = (ACTIVATION_LINE_BUFFER_DEPTH*ACTIVATION_BUFFER_BANK_COUNT*ACTIVATION_BANK_BIT_WIDTH)//8 
ACTIVATION_BUFFER_TOTAL_SIZE              = (NUMBER_OF_ACTIVATION_LINE_BUFFERS*ACTIVATION_LINE_BUFFER_DEPTH*ACTIVATION_BUFFER_BANK_COUNT*ACTIVATION_BANK_BIT_WIDTH)//8
OUTPUT_WRITER_ADDRESS_BIT_WIDTH           = np.ceil(np.log2(ACTIVATION_BUFFER_BANK_COUNT)) + np.ceil(np.log2(NUMBER_OF_ACTIVATION_LINE_BUFFERS)) + np.ceil(np.log2(ACTIVATION_LINE_BUFFER_DEPTH))
ACTIVATION_LINE_BUFFER_0_START_ADDRESS    = 0
ACTIVATION_LINE_BUFFER_1_START_ADDRESS    = ACTIVATION_LINE_BUFFER_0_START_ADDRESS + ACTIVATION_LINE_BUFFER_SIZE
ACTIVATION_LINE_BUFFER_2_START_ADDRESS    = ACTIVATION_LINE_BUFFER_1_START_ADDRESS + ACTIVATION_LINE_BUFFER_SIZE
ACTIVATION_LINE_BUFFER_3_START_ADDRESS    = ACTIVATION_LINE_BUFFER_2_START_ADDRESS + ACTIVATION_LINE_BUFFER_SIZE
ACTIVATION_LINE_BUFFER_4_START_ADDRESS    = ACTIVATION_LINE_BUFFER_3_START_ADDRESS + ACTIVATION_LINE_BUFFER_SIZE
ACTIVATION_LINE_BUFFER_5_START_ADDRESS    = ACTIVATION_LINE_BUFFER_4_START_ADDRESS + ACTIVATION_LINE_BUFFER_SIZE

SDK_FILE_NAME               = "test_neural_net" # the file name that the systemverilog simulation will generate. This file is used in the baremetal application. 
# BASE_DIRECTORY              = "/home/hasan/NVP_v1/tb/phosphene_NN_no_skips_32_pes_8_bits_64_axi/"
BASE_DIRECTORY              = "/home/hasan/NVP_v1/tb/" + out_directory_name + "/"
AXI_BYTE_ACCESS_BITS        = int(np.ceil(np.log2(axi_word_width/8)))
OUTPUT_LINE_0_START_ADDRESS = ACTIVATION_LINE_BUFFER_3_START_ADDRESS >> AXI_BYTE_ACCESS_BITS
OUTPUT_LINE_1_START_ADDRESS = ACTIVATION_LINE_BUFFER_4_START_ADDRESS >> AXI_BYTE_ACCESS_BITS
OUTPUT_LINE_2_START_ADDRESS = ACTIVATION_LINE_BUFFER_5_START_ADDRESS >> AXI_BYTE_ACCESS_BITS

f = open(out_directory_name + "/test_NN_package.sv", "w")
f2 = open(out_directory_name + "/test_NN_initial_block_code.sv", "w")
f.write(""" 
`timescale 1ns / 1ps 

package test_NN_package;

import NVP_v1_constants::*;
import test_package::*;

    localparam SDK_FILE_NAME                = "{}";
    localparam BASE_DIRECTORY               = "{}";
    localparam AXI_BYTE_ACCESS_BITS         = {};
    localparam OUTPUT_LINE_0_START_ADDRESS  = {};
    localparam OUTPUT_LINE_1_START_ADDRESS  = {};
    localparam OUTPUT_LINE_2_START_ADDRESS  = {};

    localparam ACTIVATION_ROWS              = {}; 
    localparam ACTIVATION_COLS              = {};
    localparam ACTIVATION_CHANNELS          = {};
    
""".format(
    SDK_FILE_NAME,
    BASE_DIRECTORY,
    AXI_BYTE_ACCESS_BITS,
    OUTPUT_LINE_0_START_ADDRESS,
    OUTPUT_LINE_1_START_ADDRESS,
    OUTPUT_LINE_2_START_ADDRESS,
    ROWS,
    COLS,
    max(CHANNELS, activation_precision)))

conv_output = quantized_input_mat
i = 0
ic(model)
for name, module in model.named_modules():
    if(isinstance(module,torch.nn.intrinsic.quantized.ConvReLU2d) or isinstance(module,torch.nn.quantized.Conv2d)):
        # continue
        # print(type(module))
        print(index, name, module)
        # print(module.weight.shape)

        NUMBER_OF_INPUT_ROWS       = conv_output.shape[2]
        NUMBER_OF_INPUT_COLS       = conv_output.shape[3]
        NUMBER_OF_INPUT_CHANNELS   = max(conv_output.shape[1], activation_precision)

        conv_output = module(conv_output)

        NUMBER_OF_OUTPUT_ROWS                       = conv_output.shape[2]
        NUMBER_OF_OUTPUT_COLS                       = conv_output.shape[3]
        NUMBER_OF_OUTPUT_CHANNELS                   = conv_output.shape[1]
        STRIDED_CONV                                = module.stride[0]-1
        # BIAS_ENABLE                                 = int(not (module.bias == None))
        BIAS_ENABLE                                 = 1 
        RELU_ENABLE                                 = 0 if (i==index-1) else 1
        # COMPRESS_OUTPUT                             = 0 if (i==index-1) else 1
        COMPRESS_OUTPUT                             = 1
        Q_SCALE                                     = quantization_scale_list[i]
        NUMBER_OF_KERNELS                           = module.weight().int_repr().shape[0] #module.weight.size(0)
        # NUMBER_OF_KERNELS                           = activation_precision if (NUMBER_OF_KERNELS<activation_precision) else NUMBER_OF_KERNELS
        KERNEL_K                                    = 3
        KERNEL_STEPS                                = 1 if (NUMBER_OF_KERNELS//NUMBER_OF_PES_PER_ARRAY == 0)  else NUMBER_OF_KERNELS//NUMBER_OF_PES_PER_ARRAY
        CHANNEL_STEPS                               = 1 if (NUMBER_OF_INPUT_CHANNELS//NUMBER_OF_PES_PER_ARRAY == 0)  else NUMBER_OF_INPUT_CHANNELS//NUMBER_OF_PES_PER_ARRAY
        OUTPUT_SLICES                               = NUMBER_OF_KERNELS//8 if (NUMBER_OF_KERNELS<NUMBER_OF_PES_PER_ARRAY) else NUMBER_OF_PES_PER_ARRAY//8
        OUTPUT_SLICES                               = 1 if (NUMBER_OF_KERNELS<8) else OUTPUT_SLICES
        MAX_KERNELS_AND_PES                         = NUMBER_OF_KERNELS if(NUMBER_OF_KERNELS>NUMBER_OF_PES_PER_ARRAY) else NUMBER_OF_PES_PER_ARRAY
        SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS          = (NUMBER_OF_INPUT_COLS*KERNEL_STEPS)//(STRIDED_CONV+1) + STRIDED_CONV
        NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY   = (1*KERNEL_K*NUMBER_OF_INPUT_CHANNELS*MAX_KERNELS_AND_PES*weight_precision)//weight_axi_word_width
        NUMBER_OF_BIAS_ARRAY_ENTRIES                = (NUMBER_OF_KERNELS*bias_precision_axi)//weight_axi_word_width
        NUMBER_OF_BIAS_ARRAY_ENTRIES                = 1 if(NUMBER_OF_BIAS_ARRAY_ENTRIES==0) else NUMBER_OF_BIAS_ARRAY_ENTRIES
        

        append_string = """
    localparam LAYER_{}_NUMBER_OF_KERNELS                            = {};            
    localparam LAYER_{}_STRIDED_CONV                                 = {};    
    localparam LAYER_{}_BIAS_ENABLE                                  = {};    
    localparam LAYER_{}_RELU_ENABLE                                  = {};    
    localparam LAYER_{}_COMPRESS_OUTPUT                              = {};    
    localparam LAYER_{}_Q_SCALE                                      = {};    
    localparam LAYER_{}_KERNEL_STEPS                                 = {};    
    localparam LAYER_{}_CHANNEL_STEPS                                = {};        
    localparam LAYER_{}_OUTPUT_SLICES                                = {};        
    localparam LAYER_{}_MAX_KERNELS_AND_PES                          = {};            
    localparam LAYER_{}_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS           = {};                            
    localparam LAYER_{}_NUMBER_OF_OUTPUT_COLS                        = {};                
    localparam LAYER_{}_NUMBER_OF_OUTPUT_CH                          = {};            
    localparam LAYER_{}_NUMBER_OF_OUTPUT_ROWS                        = {};                
    localparam LAYER_{}_KERNEL_K                                     = {};
    localparam LAYER_{}_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY    = {};
    localparam LAYER_{}_NUMBER_OF_BIAS_ARRAY_ENTRIES                 = {};                                        
        """.format(
            i, NUMBER_OF_KERNELS,
            i, STRIDED_CONV,
            i, BIAS_ENABLE,
            i, RELU_ENABLE,
            i, COMPRESS_OUTPUT,
            i, Q_SCALE,
            i, KERNEL_STEPS,
            i, CHANNEL_STEPS,
            i, OUTPUT_SLICES,
            i, MAX_KERNELS_AND_PES,
            i, SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS,
            i, NUMBER_OF_OUTPUT_COLS,
            i, NUMBER_OF_OUTPUT_CHANNELS,
            i, NUMBER_OF_OUTPUT_ROWS,
            i, KERNEL_K,
            i, NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY,
            i, NUMBER_OF_BIAS_ARRAY_ENTRIES
        )

        if(i==0):
            append_string_2 = """
    neural_network_layer #(
        .LAYER_ID                            ({}),
        .BASE_DIRECTORY                      (BASE_DIRECTORY),
        .AXI_BUS_DATA_BIT_WIDTH              (AXI_BUS_BIT_WIDTH),
        .AXI_BUS_ADDRESS_WIDTH               (AXI_BUS_ADDRESS_WIDTH),
        .WEIGHT_AXI_BUS_DATA_BIT_WIDTH       (WEIGHT_AXI_BUS_BIT_WIDTH),
        .WEIGHT_AXI_BUS_ADDRESS_WIDTH        (WEIGHT_AXI_BUS_ADDRESS_WIDTH),
        .CONTROL_AXI_DATA_WIDTH              (CONTROL_AXI_DATA_WIDTH),
        .CONTROL_AXI_ADDR_WIDTH              (CONTROL_AXI_ADDR_WIDTH),
        .export_file_name                    (SDK_FILE_NAME),
        .AXI_BYTE_ACCESS_BITS                (AXI_BYTE_ACCESS_BITS),
        .OUTPUT_LINE_0_START_ADDRESS         (OUTPUT_LINE_0_START_ADDRESS),
        .OUTPUT_LINE_1_START_ADDRESS         (OUTPUT_LINE_1_START_ADDRESS),
        .OUTPUT_LINE_2_START_ADDRESS         (OUTPUT_LINE_2_START_ADDRESS),
        .INPUT_NUMBER_OF_COLS                (ACTIVATION_COLS),            
        .INPUT_NUMBER_OF_CH                  (ACTIVATION_CHANNELS),        
        .INPUT_NUMBER_OF_ROWS                (ACTIVATION_ROWS),            
        .STRIDED_CONV                        (LAYER_{}_STRIDED_CONV),
        .BIAS_ENABLE                         (LAYER_{}_BIAS_ENABLE),
        .RELU_ENABLE                         (LAYER_{}_RELU_ENABLE),
        .COMPRESS_OUTPUT                     (LAYER_{}_COMPRESS_OUTPUT),
        .Q_SCALE                             (LAYER_{}_Q_SCALE),
        .KERNEL_STEPS                        (LAYER_{}_KERNEL_STEPS),
        .CHANNEL_STEPS                       (LAYER_{}_CHANNEL_STEPS),
        .OUTPUT_SLICES                       (LAYER_{}_OUTPUT_SLICES),
        .SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS  (LAYER_{}_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS),
        .NUMBER_OF_WEIGHT_ARRAY_ENTRIES      (LAYER_{}_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY),
        .NUMBER_OF_BIAS_ARRAY_ENTRIES        (LAYER_{}_NUMBER_OF_BIAS_ARRAY_ENTRIES),
        .NUMBER_OF_OUTPUT_COLS               (LAYER_{}_NUMBER_OF_OUTPUT_COLS),            
        .NUMBER_OF_OUTPUT_CH                 (LAYER_{}_NUMBER_OF_OUTPUT_CH),            
        .NUMBER_OF_OUTPUT_ROWS               (LAYER_{}_NUMBER_OF_OUTPUT_ROWS)         
    ) layer_{};
            """.format(i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i)

            append_string_3 = """
    $display("execute layer_{}");
    layer_{}.execute(
            .clk                    (clk),
            .data_bus               (data_bus),
            .weight_bus             (weight_bus),
            .control_bus            (control_bus),
            .next_command_interrupt (next_command_interrupt),
            .output_line_stored     (output_line_stored)
        );
    layer_{}.validate_outputs();

        """.format(i,i,i)

        else:
            append_string_2 = """
    neural_network_layer #(
        .LAYER_ID                            ({}),
        .BASE_DIRECTORY                      (BASE_DIRECTORY),
        .AXI_BUS_DATA_BIT_WIDTH              (AXI_BUS_BIT_WIDTH),
        .AXI_BUS_ADDRESS_WIDTH               (AXI_BUS_ADDRESS_WIDTH),
        .WEIGHT_AXI_BUS_DATA_BIT_WIDTH       (WEIGHT_AXI_BUS_BIT_WIDTH),
        .WEIGHT_AXI_BUS_ADDRESS_WIDTH        (WEIGHT_AXI_BUS_ADDRESS_WIDTH),
        .CONTROL_AXI_DATA_WIDTH              (CONTROL_AXI_DATA_WIDTH),
        .CONTROL_AXI_ADDR_WIDTH              (CONTROL_AXI_ADDR_WIDTH),
        .export_file_name                    (SDK_FILE_NAME),
        .AXI_BYTE_ACCESS_BITS                (AXI_BYTE_ACCESS_BITS),
        .OUTPUT_LINE_0_START_ADDRESS         (OUTPUT_LINE_0_START_ADDRESS),
        .OUTPUT_LINE_1_START_ADDRESS         (OUTPUT_LINE_1_START_ADDRESS),
        .OUTPUT_LINE_2_START_ADDRESS         (OUTPUT_LINE_2_START_ADDRESS),
        .INPUT_NUMBER_OF_COLS                (LAYER_{}_NUMBER_OF_OUTPUT_COLS),            
        .INPUT_NUMBER_OF_CH                  (LAYER_{}_NUMBER_OF_OUTPUT_CH),        
        .INPUT_NUMBER_OF_ROWS                (LAYER_{}_NUMBER_OF_OUTPUT_ROWS),            
        .STRIDED_CONV                        (LAYER_{}_STRIDED_CONV),
        .BIAS_ENABLE                         (LAYER_{}_BIAS_ENABLE),
        .RELU_ENABLE                         (LAYER_{}_RELU_ENABLE),
        .COMPRESS_OUTPUT                     (LAYER_{}_COMPRESS_OUTPUT),
        .Q_SCALE                             (LAYER_{}_Q_SCALE),
        .KERNEL_STEPS                        (LAYER_{}_KERNEL_STEPS),
        .CHANNEL_STEPS                       (LAYER_{}_CHANNEL_STEPS),
        .OUTPUT_SLICES                       (LAYER_{}_OUTPUT_SLICES),
        .SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS  (LAYER_{}_SINGLE_ROW_TOTAL_NUMBER_OF_OUTPUTS),
        .NUMBER_OF_WEIGHT_ARRAY_ENTRIES      (LAYER_{}_NUMBER_OF_WEIGHT_ENTRIES_PER_WEIGHT_ARRAY),
        .NUMBER_OF_BIAS_ARRAY_ENTRIES        (LAYER_{}_NUMBER_OF_BIAS_ARRAY_ENTRIES),
        .NUMBER_OF_OUTPUT_COLS               (LAYER_{}_NUMBER_OF_OUTPUT_COLS),            
        .NUMBER_OF_OUTPUT_CH                 (LAYER_{}_NUMBER_OF_OUTPUT_CH),            
        .NUMBER_OF_OUTPUT_ROWS               (LAYER_{}_NUMBER_OF_OUTPUT_ROWS)         
    ) layer_{};
            """.format(i,i-1,i-1,i-1,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i)

            append_string_3 = """
    $display("execute layer_{}");
    layer_{}.row_i = layer_{}.output_row_i;
    // layer_{}.row_i = layer_{}.ground_truth_output_row_i;
    layer_{}.row_i_number_of_entries = layer_{}.output_row_i_number_of_entries;
    layer_{}.execute(
            .clk                    (clk),
            .data_bus               (data_bus),
            .weight_bus             (weight_bus),
            .control_bus            (control_bus),
            .next_command_interrupt (next_command_interrupt),
            .output_line_stored     (output_line_stored)
        );
    layer_{}.validate_outputs();

        """.format(i,i,i-1,i,i-1,i,i-1,i,i)

        f.write(append_string)
        f.write(append_string_2)
        f2.write(append_string_3)

        i+=1

f.write(""" 

endpackage
    
""")

            
f.close()
f2.close()
# print(model)

# Print MACs and model size (using https://github.com/Lyken17/pytorch-OpCounter)
# from thop import profile
# input = torch.randn(1, 1, 128, 256)
# macs, params = profile(model, inputs=(input, ))
# print("macs, params")
# print(macs, params)

# ic(qweight_scale)
# ic(q_input_activation_scale)
print("seed: ",seed)
ic(quantization_scale_list)
# exit()


# int8_metrics = PTQ.evaluate_model(cfg, quantized_encoder, decoder, simulator, valloader, device=cpu_device, quantized=True, criterion=None, visualize=5, savefig=False)
# ic(int8_metrics)


def evaluate_hw_simulation_model(cfg, encoder, decoder, simulator, test_loader, device, quantized=False, criterion=None, visualize=None, savefig=False):

    encoder.eval()
    encoder.to(device)
    decoder.eval()
    decoder.to(device)
    
    # Forward pass (validation set)
    # image,label = next(iter(test_loader))
    # image.to(device)
    # label.to(device)

    # image = image[0]
    # label = label[0]
    # image = image[None, :]
    # label = label[None, :]
    
    quantized_input_mat = encoder.quant(image)
    conv_output = quantized_input_mat
    q_input_activation_scale = quantized_encoder.quant.scale.item()
    q_output_activation_scale = q_input_activation_scale

    # ic(device)
    # ic(image.device)
    index = 0
    with torch.no_grad():
        stimulation = encoder(image)
        for name, module in model.named_modules():
            if(isinstance(module,torch.nn.intrinsic.quantized.ConvReLU2d)  or isinstance(module,torch.nn.quantized.Conv2d)):
                ic(name, module)
                weights = module.weight().int_repr()
                bias = module.bias()
                q_input_activation_scale = q_output_activation_scale
                qweight_scale = module.weight().q_scale()
                # qweight_zero_point = module.weight().q_zero_point()
                q_output_activation_scale = module.scale
                q_output_activation_zero_point = module.zero_point
                ic(q_output_activation_zero_point)
                qbias_scale = qweight_scale*q_input_activation_scale
                scale = (qweight_scale*q_input_activation_scale) / q_output_activation_scale
                quantization_scale_list.append(round((scale)*(2**quantization_scale_precision)))
                index += 1
                bias   = torch.nn.Parameter(torch.clip(torch.floor(bias/qbias_scale).float(),  min=qbias_min, max=qbias_max)) 
                if(index==1):
                    simulated_conv_output = torch.nn.functional.conv2d(input=conv_output.int_repr().float(), weight=weights.float(), bias=bias, stride=module.stride[0], padding=module.padding) 
                else:
                    simulated_conv_output = torch.nn.functional.conv2d(input=conv_output.float(), weight=weights.float(), bias=bias, stride=module.stride[0], padding=module.padding) 
                if(isinstance(module,torch.nn.intrinsic.quantized.ConvReLU2d)):
                    simulated_conv_output = torch.nn.functional.relu(simulated_conv_output)
                    ic("inside relu")
                    simulated_conv_output = (simulated_conv_output*quantization_scale_list[index-1])  // (2**quantization_scale_precision)
                else:
                    ic(simulated_conv_output.shape)
                    # ic(simulated_conv_output.detach().numpy())
                    ic(simulated_conv_output.detach().numpy()[0][0][0])
                    # ic(simulated_conv_output.detach().numpy()[0][0][15])  
                    simulated_conv_output = (simulated_conv_output*quantization_scale_list[index-1])  / (2**quantization_scale_precision)
                    simulated_conv_output = torch.floor(simulated_conv_output)
                # simulated_conv_output = (simulated_conv_output*quantization_scale_list[index-1])  // (2**quantization_scale_precision)
                conv_output = simulated_conv_output
                conv_output_np = conv_output.detach().numpy()
                
                # ic(conv_output_np)
        stimulation_test      = conv_output
        # ic(encoder)
        # ic(stimulation)
        # ic(stimulation_test)
        # ic(stimulation_test.detach().numpy()[0][0][0])  
        for i in range(stimulation_test.shape[2]):
            ic(i)
            print(stimulation_test.detach().numpy()[0][0][i])  
        stimulation_test = torch.tanh(stimulation_test)
        # ic(stimulation_test)
        # ic(stimulation_test - stimulation)
        # ic(q_output_activation_zero_point)
        # stimulation_test = torch.tanh(stimulation_test)
        # ic(stimulation_test)
        # stimulation_test = torch.tanh(stimulation_test)
        # ic(stimulation_test)
        # stimulation_test = torch.tanh(stimulation_test)
        # ic(stimulation_test)


        # ic(stimulation.detach().numpy())        
        # ic(stimulation_test.detach().numpy())     
        # ic(stimulation_test.detach().numpy() - stimulation.detach().numpy())   
        # ic(torch.sign(stimulation).detach().numpy())
        # exit()
        # phosphenes      = simulator(stimulation_test)


        # stimulation_test = stimulation_test + torch.sign(stimulation_test).detach() - stimulation_test.detach() # (self-through estimator)
        # stimulation_test = .5*(stimulation_test+1)
        # ic(stimulation_test.detach().numpy())     
        # from torchvision.utils import save_image 
        # string = 'test_phosphene_image.png'
        # save_image(stimulation_test, string)
        
        # stimulation_test = 255*stimulation_test
        # # ic(stimulation_test.detach().numpy())    
        # string = 'test_phosphene_image_int.png'
        # save_image(stimulation_test, string)
        # exit()

        # ic(stimulation_test.detach().numpy())     
        # torch.nn.functional.upsample(stimulation_test, mode="nearest", scale_factor=8)
        # phosphenes = stimulation_test

        phosphenes      = simulator(stimulation_test)
        reconstruction  = decoder(phosphenes)    

    # Visualize results
    if visualize is not None:
        n_figs = 4 if cfg.reconstruction_loss == 'boundary' else 3
        n_examples = visualize
        plt.figure(figsize=(n_figs,n_examples),dpi=200)

        for i in range(n_examples):
            plt.subplot(n_examples,n_figs,n_figs*i+1)
            plt.imshow(image[i].squeeze().cpu().numpy(),cmap='gray')
            plt.axis('off')
            plt.subplot(n_examples,n_figs,n_figs*i+2)
            plt.imshow(phosphenes[i].squeeze().cpu().numpy(),cmap='gray')
            plt.axis('off')
            plt.subplot(n_examples,n_figs,n_figs*i+3)
            plt.imshow(reconstruction[i][0].squeeze().cpu().numpy(),cmap='gray')
            plt.axis('off')
            if n_figs > 3:
                plt.subplot(n_examples,n_figs,n_figs*i+4)
                plt.imshow(label[i].squeeze().cpu().numpy(),cmap='gray')
                plt.axis('off')
        if savefig:
            if(quantized):
                plt.savefig(os.path.join(cfg.savedir,cfg.model_name+'eval_quantized.png'))
            else:
                plt.savefig(os.path.join(cfg.savedir,cfg.model_name+'eval.png'))
        plt.show()

    # Calculate performance metrics
    im_pairs = [[im.squeeze().cpu().numpy(),trg.squeeze().cpu().numpy()] for im,trg in zip(image,reconstruction)]
    
    if cfg.reconstruction_loss == 'boundary':
        metrics=pd.Series() #TODO
    else:
        mse = [mean_squared_error(*pair) for pair in im_pairs]
        ssim = [structural_similarity(*pair, gaussian_weigths=True) for pair in im_pairs]
        psnr = [peak_signal_noise_ratio(*pair) for pair in im_pairs]
        metrics=pd.Series({'mse':np.mean(mse),
                           'ssim':np.mean(ssim),
                           'psnr':np.mean(psnr)})
    return metrics


int8_metrics_hw_simulation  = evaluate_hw_simulation_model(cfg, quantized_encoder, decoder, simulator, valloader, device=cpu_device, quantized=True, criterion=None, visualize=1, savefig=False)
ic(int8_metrics_hw_simulation)