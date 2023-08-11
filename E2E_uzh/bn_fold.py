import torch.nn as nn
import os 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torchvision
import torch
import utils,training,model, NVP_v1_model
import local_datasets
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader 

def fuse_single_conv_bn_pair(block1, block2):
    if isinstance(block1, nn.BatchNorm2d) and isinstance(block2, nn.Conv2d):
        m = block1
        conv = block2
        
        bn_st_dict = m.state_dict()
        conv_st_dict = conv.state_dict()

        # BatchNorm params
        eps = m.eps
        mu = bn_st_dict['running_mean']
        var = bn_st_dict['running_var']
        gamma = bn_st_dict['weight']

        if 'bias' in bn_st_dict:
            beta = bn_st_dict['bias']
        else:
            beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

        # Conv params
        W = conv_st_dict['weight']
        if 'bias' in conv_st_dict:
            bias = conv_st_dict['bias']
        else:
            bias = torch.zeros(W.size(0)).float().to(gamma.device)

        denom = torch.sqrt(var + eps)
        b = beta - gamma.mul(mu).div(denom)
        A = gamma.div(denom)
        bias *= A
        A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

        W.mul_(A)
        bias.add_(b)

        conv.weight.data.copy_(W)

        if conv.bias is None:
            conv.bias = torch.nn.Parameter(bias)
        else:
            conv.bias.data.copy_(bias)
            
        return conv
        
    else:
        return False
    
def fuse_bn_recursively(model):
    previous_name = None
    
    for module_name in model._modules:
        previous_name = module_name if previous_name is None else previous_name # Initialization
        
        conv_fused = fuse_single_conv_bn_pair(model._modules[module_name], model._modules[previous_name])
        if conv_fused:
            model._modules[previous_name] = conv_fused
            model._modules[module_name] = nn.Identity()
            
        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])
            
        previous_name = module_name

    return model



if __name__ == '__main__':
    import argparse
    import pandas as pd
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_name", type=str, default="demo_model",
                    help="model name")
    ap.add_argument("-dir", "--savedir", type=str, default="./out/demo",
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
    ap.add_argument("-dev", "--device", type=str, default="cuda:0",
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
    ap.add_argument("-nvp", "--NVP_v1", action='store_true', default=False)
    

    cfg = pd.Series(vars(ap.parse_args()))
    print(cfg)
    models, dataset, optimization, train_settings = training.initialize_components(cfg)
    encoder = models['encoder']
    print(encoder)
    fuse_bn_recursively(encoder)
    print(encoder)

