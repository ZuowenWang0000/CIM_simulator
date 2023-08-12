import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import time
import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import Dataset, DataLoader 
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
import utils 
import NVP_v1_model, training_nvp
import local_datasets
from icecream import ic
import argparse
import pandas as pd
from bn_fold import fuse_bn_recursively


def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform) 
    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!
    test_set = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=num_workers)

    return train_loader, test_loader

def evaluate_model(cfg, encoder, decoder, simulator, test_loader, device, quantized=False, criterion=None, visualize=None, savefig=False):

    encoder.eval()
    encoder.to(device)
    decoder.eval()
    decoder.to(device)
    
    # Forward pass (validation set)
    image,label = next(iter(test_loader))
    
    image.to(device)
    label.to(device)
    print(f"image dtype:{image.dtype}, label dtype:{label.dtype}, device:{device}")

    ic(device)
    ic(image.device)
    with torch.no_grad():
        stimulation = encoder(image)
        phosphenes  = simulator(stimulation)
        reconstruction = decoder(phosphenes)    

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
        # psnr = [peak_signal_noise_ratio(*pair) for pair in im_pairs]
        metrics=pd.Series({'mse':np.mean(mse),
                           'ssim':np.mean(ssim),
                        #    'psnr':np.mean(psnr)
                           })
    return metrics

def calibrate_model(model, loader, device=torch.device("cpu:0")):

    model.to(device)
    model.eval()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)

def measure_inference_latency(model,
                              device,
                              input_size=(1, 3, 32, 32),
                              num_samples=100,
                              num_warmups=10):

    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave

def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)

def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model

def save_torchscript_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)

def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model

def create_model(num_classes=10):

    # The number of channels in ResNet18 is divisible by 8.
    # This is required for fast GEMM integer matrix multiplication.
    # model = torchvision.models.resnet18(pretrained=False)
    model = resnet18(num_classes=num_classes, pretrained=False)

    # We would use the pretrained ResNet18 as a feature extractor.
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Modify the last FC layer
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 10)

    return model

def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1,3,32,32)):

    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False

    return True

def quantize_and_evaluate(cfg, visualize=None, savefig=False):

    random_seed = 0
    num_classes = 10
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")
    
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

    encoder.to(cpu_device)
    # Make a copy of the encoder for layer fusion

    #### PTQ
    fused_encoder = copy.deepcopy(encoder)

    encoder.eval()
    # The encoder has to be switched to evaluation mode before any layer fusion.
    # Otherwise the quantization will not work correctly.
    fused_encoder.eval()
    

    # for name, module in fused_encoder.named_modules():
    #     print(name)
    # exit()
    for module_name, module in fused_encoder.named_children():
        ic(module_name)
        for basic_block_name, basic_block in module.named_children():
            ic(basic_block)
            if(isinstance(basic_block, NVP_v1_model.convBlock)):
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"]], inplace=True)  
            if(isinstance(basic_block, NVP_v1_model.ResidualBlock)):
                torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"]], inplace=True)  

    # # Print FP32 encoder.
    ic(encoder)
    # # Print fused encoder.
    ic(fused_encoder)

    fused_encoder = copy.deepcopy(fused_encoder)

    # encoder and fused encoder should be equivalent.
    # assert model_equivalence(model_1=encoder, model_2=fused_encoder, device=cpu_device, rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1,1,32,32)), "Fused encoder is not equivalent to the original encoder!"
    # exit()


    quantization_config = torch.quantization.QConfig(
    activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine), 
    weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric))
    print(quantization_config)

    # Prepare the encoder for static quantization. This inserts observers in
    # the encoder that will observe activation tensors during calibration.
    quantized_encoder = NVP_v1_model.Quantized_E2E_Encoder(model_fp32=fused_encoder)
    # Using un-fused encoder will fail. Because there is no quantized layer implementation for a single batch normalization layer.
    # Select quantization schemes from 
    # https://pytorch.org/docs/stable/quantization-support.html
    # quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    # Custom quantization configurations
    # quantization_config = torch.quantization.default_qconfig

    
    quantized_encoder.qconfig = quantization_config
    
    # Print quantization configurations
    # print(quantized_encoder.qconfig)

    torch.quantization.prepare(quantized_encoder, inplace=True)

    # Use training data for calibration.
    print("calibrate model")
    calibrate_model(model=quantized_encoder, loader=trainloader, device=cpu_device)
    print("calibrated!")
    quantized_encoder = torch.quantization.convert(quantized_encoder, inplace=True)
    print("quantized")
    quantized_encoder.eval()

    # Print quantized model.
    print(quantized_encoder)
    # print(encoder.conv1)
    # print(quantized_encoder.model_fp32.conv1)

    model_dir = cfg.savedir
    model_filename = cfg.model_name+'_best_encoder.pth'
    quantized_model_filename = cfg.model_name+'_best_encoder_quantized.pth'
    quantized_jit_model_filename = cfg.model_name+'_jit_best_encoder_quantized.pth'
    model_filepath = os.path.join(model_dir, model_filename)
    quantized_model_filepath = os.path.join(model_dir, quantized_model_filename)
    quantized_jit_model_filepath = os.path.join(model_dir, quantized_jit_model_filename)
    # Save quantized encoder. 
    torch.save(quantized_encoder.state_dict(), quantized_model_filepath)
    save_torchscript_model(model=quantized_encoder, model_dir=model_dir, model_filename=quantized_jit_model_filename)

    # Load quantized encoder.
    quantized_jit_encoder = load_torchscript_model(model_filepath=quantized_jit_model_filepath, device=cpu_device)

    fp32_metrics = evaluate_model(cfg, encoder, decoder, simulator, valloader, device=cpu_device, quantized=False, criterion=None, visualize=visualize, savefig=savefig)
    print("FP32 evaluation: ", fp32_metrics)
    int8_metrics = evaluate_model(cfg, quantized_jit_encoder, decoder, simulator, valloader, device=cpu_device, quantized=True, criterion=None, visualize=visualize, savefig=savefig)
    print("INT8 evaluation: ", int8_metrics)

    # Skip this assertion since the values might deviate a lot.
    # assert model_equivalence(model_1=model, model_2=quantized_jit_model, device=cpu_device, rtol=1e-01, atol=1e-02, num_tests=100, input_size=(1,3,32,32)), "Quantized model deviates from the original model too much!"

    fp32_cpu_inference_latency = measure_inference_latency(model=encoder, device=cpu_device, input_size=(1,1,32,32), num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(model=quantized_encoder, device=cpu_device, input_size=(1,1,32,32), num_samples=100)
    int8_jit_cpu_inference_latency = measure_inference_latency(model=quantized_jit_encoder, device=cpu_device, input_size=(1,1,32,32), num_samples=100)
    fp32_gpu_inference_latency = measure_inference_latency(model=encoder, device=cuda_device, input_size=(1,1,32,32), num_samples=100)
    int8_gpu_inference_latency = measure_inference_latency(model=quantized_encoder, device=cuda_device, input_size=(1,1,32,32), num_samples=100)
    
    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    print("FP32 CUDA Inference Latency: {:.2f} ms / sample".format(fp32_gpu_inference_latency * 1000))
    print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
    print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))
    print("INT8 CUDA Inference Latency: {:.2f} ms / sample".format(int8_gpu_inference_latency * 1000))



class testBlock(nn.Module):
    def __init__(self, n_channels, stride=1):
        super(testBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels,kernel_size=3, stride=1,padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        return out


if __name__ == '__main__':

    
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_name", type=str, default="demo_model_imc_sim_ptq",
                    help="model name")
    ap.add_argument("-dir", "--savedir", type=str, default="nvp_NN_character_imc_sim/",
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
    cfg = pd.Series(vars(ap.parse_args()))

    print(cfg)

    metrics = quantize_and_evaluate(cfg, 5, savefig=True)
    print(metrics)
    