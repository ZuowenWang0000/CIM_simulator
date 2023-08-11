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
ap.add_argument("-n", "--batch_size", type=int, default=100,
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
out_directory_name = "exported_compiled_images"
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

max_row_len = 0
first_set = True

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
def save_compressed_mats(compressed_mats, directory, image_index, axi_word_width, activation_precision, max_row_len, first_set):
    row = 0

    # with open(directory+"/test_compressed_images.txt", "a") as f:
    # f.write(b"\n")
    f = open(directory+"/test_compressed_images.h", "a")
    f.write(""" 
uint64_t img_ptr_{}[] = {{ 
    """.format(image_index))
    # numpy.savetxt(f, a)
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

        if (first_set):
            first_set = False
            max_row_len = len(compressed_mat_input_buffer)
        else:
            if(max_row_len < len(compressed_mat_input_buffer)):
                max_row_len = len(compressed_mat_input_buffer)
                ic(max_row_len)
        for i in range(26 - len(compressed_mat_input_buffer)):
            compressed_mat_input_buffer.append("0000000000000000")
        # ic(len(compressed_mat_input_buffer))
        # np.savetxt(f, compressed_mat_input_buffer, fmt='%s', delimiter=',', newline=',\n')
        # np.savetxt(f, compressed_mat_input_buffer, fmt='%s', newline=',\n')
        np.savetxt(f, compressed_mat_input_buffer, fmt='0x%s,')
    f.write(""" 
    };
    """)
    f.close()
    return max_row_len, first_set

#############
# Compress and save input activations
#############

f = open(out_directory_name+"/test_compressed_images.h", "w")
f.write(""" 
#ifndef SRC_TEST_COMPRESSED_IMAGES_H_ 
#define SRC_TEST_COMPRESSED_IMAGES_H_ 
""")
f.close()
f2 = open(out_directory_name + "/test_images.h", "w")
f2.write(""" 
#ifndef SRC_TEST_IMAGES_H_ 
#define SRC_TEST_IMAGES_H_ 
""")

# ic(valloader)
# exit()
images,label = next(iter(valloader))
NUM_IMAGES_TO_EXPORT = images.shape[0]
for i in range(NUM_IMAGES_TO_EXPORT):
    
    image = images[i]
    image = image[None, :]
    image.to(cfg.device)
    # ## Visualize
    # import matplotlib.pyplot as plt
    # import utils
    # plt.figure(figsize=(10,10),dpi=50)
    # utils.plot_images(image)

    quantized_input_mat = model.quant(image)
    quantized_input_mat_int = quantized_input_mat.int_repr()
    
    num_input_channel_sub_tiles = max(i_channels//activation_precision, 1)
    if(i_channels<activation_precision):
        channels_per_sub_tile = i_channels
    else:
        channels_per_sub_tile = activation_precision
    compressed_mats = sm_compression(quantized_input_mat_int, i_rows, i_cols, num_input_channel_sub_tiles, channels_per_sub_tile)
    # input_activation_dir = out_directory_name + "/input_activations_"+ str(i)
    # if not os.path.exists(input_activation_dir):
    #     os.mkdir(input_activation_dir)
    #     print("Directory " , input_activation_dir ,  " Created ")
    # else:    
    #     print("Directory " , input_activation_dir ,  " already exists")
    max_row_len, first_set = save_compressed_mats(compressed_mats, out_directory_name, i, axi_word_width, activation_precision, max_row_len, first_set)

#  u8 layer_0_weight_array_1[48] = { 
#  	0x49, 
#  	0x00, 
#  }; 

#  uint64_t total_number_of_input_activations_entries = 2407; 
#  uint64_t input_activations[2407] = {


    f2.write(""" 
    u8 img_{}[16384] = {{ 
    """.format(i))
    
    for row_index in range(i_rows):
        for col_index in range(i_cols):
            f2.write(""" 
                {},""".format(quantized_input_mat_int[0,0,row_index,col_index]))

    f2.write(""" 
    };
    """)

ic(max_row_len)

f = open(out_directory_name+"/test_compressed_images.h", "a")
f.write(""" 
#endif /* SRC_TEST_COMPRESSED_IMAGES_H_ */ 
""")

f.write(""" 
uint64_t* compressed_test_image[] = {
    """)
for i in range(NUM_IMAGES_TO_EXPORT):
    f.write(""" 
        img_ptr_{}, 
    """.format(i))
f.write(""" 
    };
    """)

f.close()


f2.write(""" 
u8* test_image[] = {
    """)
for i in range(NUM_IMAGES_TO_EXPORT):
    f2.write(""" 
        img_{}, 
    """.format(i))
f2.write(""" 
    };
    """)
f2.write(""" 
#endif /* SRC_TEST_IMAGES_H_ */ 
""")
f2.close()

ic(NUM_IMAGES_TO_EXPORT)