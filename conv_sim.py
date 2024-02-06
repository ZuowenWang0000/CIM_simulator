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
from functionals.util import format_large_number, format_small_number
from models.simple_cnn import SimpleNVPCNN
from local_datasets import ADE_Dataset, Character_Dataset
from torch.utils.data import Dataset, DataLoader 



def main(args):
    config = yaml.load(open('kt_qy_imc.yaml', 'r'), Loader=yaml.FullLoader)

    # add args to config dictionary
    for key, value in vars(args).items():
        config[key] = value

    print(config)


    bias = torch.load('bias.pt')
    stride = torch.load('stride.pt')
    padding = torch.load('padding.pt')
    groups = torch.load('groups.pt')
    dilation = torch.load('dilation.pt')

    I_NL = pd.read_excel('data_I-nonlinearity.xlsx').to_numpy()[:, 1]

    I_NL = dict(enumerate(I_NL.flatten(), 0))

    model = SimpleNVPCNN(config, I_NL)
    setattr(model.conv7_a, "weight", np.zeros(10))
    print(model.conv7_a.weight.shape)
    # torch load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    # iterate through the checkpoint dictionary and load the model
    # for key, value in checkpoint.items():
    #     print(key)
        # print(value.shape)
        # if 'conv' in key:
        #     if 'weight' in key:
        #         # get the first numerical value in the key
        #         layer_num = int(''.join(filter(str.isdigit, key)))
        #         # set the corresponding weight
        #         eval("model.conv{}")
        #     if 'bias' in key:
        #         setattr(model, key, value)

    # print(checkpoint['model.0.conv1.weight'].shape)

    for key, value in model.__dict__.items():
        print(key)


    # print the attributes of the model
    for key, value in checkpoint.items():
        print(key)
        split_key = key.split('.')

        if len(split_key) == 1:
            continue
        elif len(split_key) >3:
            layer_num = int(split_key[1])
            layer_type = split_key[2]
            weight_or_bias = split_key[3]
            wob = checkpoint[f'model.{layer_num}.{layer_type}.{weight_or_bias}']
        else:
            layer_num = int(split_key[1])
            weight_or_bias = split_key[2]
            wob = checkpoint[f'model.{layer_num}.{weight_or_bias}']

        if layer_num < 4:
            setattr(eval(f"model.{layer_type[:-1]}{layer_num+1}"), f"{weight_or_bias}", wob)
            print(f"loaded: {key}")
        elif (layer_num >= 4) and (layer_num < 8):
            if layer_type == 'conv1':
                setattr(eval(f"model.{layer_type[:-1]}{layer_num+1}_a"), f"{weight_or_bias}", wob)
            elif layer_type == 'conv2':
                setattr(eval(f"model.{layer_type[:-1]}{layer_num+1}_b"), f"{weight_or_bias}", wob)
            print(f"loaded: {key}")
        elif layer_num == 8:
            # print(layer_type[:-1])
            setattr(eval(f"model.{layer_type[:-1]}{layer_num+1}"), f"{weight_or_bias}", wob)
            print(f"loaded: {key}")
        elif layer_num == 9:
            setattr(eval(f"model.{layer_num+1}"), f"{weight_or_bias}", wob)
            print(f"loaded: {key}")



    quit()
    # x = torch.load('x.pt',map_location='cuda:0').cpu().numpy()
    # load data with local_dataset.py

    # Dataset
    dataset = dict()
    # trainset = local_datasets.Character_Dataset(device=cfg.device, imsize = (input_image_size,input_image_size))
    valset = Character_Dataset(device='cpu',validation = True, imsize = (128, 128)) 

    # dataset['trainloader'] = DataLoader(trainset,batch_size=int(cfg.batch_size),shuffle=True)
    batch_size = 1
    dataset['valloader'] = DataLoader(valset,batch_size=int(batch_size),shuffle=False)

    weight =  torch.load('w.pt',map_location='cuda:0').cpu().numpy()
    stride = stride[0]

    # todo : acc_length is what analog sum length? then it should be 8 
    acc_length = 64

    for id, (x, y) in enumerate(dataset['valloader']):
        print(f"processing {id}th image, shape of x:{x.shape}")
        input_shape = (1,3,128,128)

        batch = x.shape[0]
        ch = x.shape[1]
        feature_size = int(x.shape[2]/stride)
        ks = weight.shape[2]
        kernel_num = weight.shape[0]
        round = int(np.ceil(ch/acc_length))
        # print(f"shape of x:{x.shape} shape of weight:{weight.shape}")
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

        if config['logging_control']['consider_analog_value_dependency']:
            output_features, layer_total_latency, layer_total_energy, total_ops = model.forward(x)
        else:
            output_features, layer_total_latency, layer_total_energy, total_ops = model.forward(input_shape)

        print(f"input feature size:{input_shape}")
        print(f"output feature size:{output_features.shape}")


        simple_column_eval = True
        if simple_column_eval:
            num_banks = config['basic_paras']['num_banks']
            column_per_bank = config['basic_paras']['num_columns']
            total_columns = num_banks * column_per_bank
            total_latency = (model.total_latency/1.0e6)/total_columns
            print(f"total latency :{format_large_number(total_latency)}s")
            print(f"total ops :{format_large_number(model.total_ops)}")

            ops_per_second = model.total_ops/(model.total_latency*1e-6)*total_columns
            print(f"ops per second:{format_large_number(ops_per_second)}")
            # print(f"ops per second:{ops_per_second}")

            print(f"IMC power consumption:\
                {format_small_number(ops_per_second/(config['energy']['peak_energy_efficient']*1.0e12))}W")

        if config['logging_control']['consider_analog_value_dependency']:
            break

if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_name", type=str, default="demo_model_imc_sim",
                    help="model name")
    ap.add_argument("-ckp", "--checkpoint", type=str, default="/home/zuowen_temp/Desktop/e2e_uzh/E2E_model_lighten_uzh/viseon_SKU_new_repo/out/nov30/exp5_uzh_model/exp5_B_S4a_resnet_noskip_best_encoder.pth",
                    help="checkpoint to resume from")
    args = ap.parse_args()

    print(args)

    main(args)



# print(f"total energy:{model.total_energy}")

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



