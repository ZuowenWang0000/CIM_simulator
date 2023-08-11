import csv
import os
## Import statements

import os
import numpy as np
import matplotlib.pyplot as plt
import math

# Local dependencies
import utils ,model
import NVP_v1_model
import local_datasets
from icecream import ic
import shutil

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader 


from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio


class CustomLoss(object):
    def __init__(self, recon_loss_type='mse',recon_loss_param=None, stimu_loss_type=None, kappa=0, device='cpu'):
        """Custom loss class for training end-to-end model with a combination of reconstruction loss and sparsity loss
        reconstruction loss type can be either one of: 'mse' (pixel-intensity based), 'vgg' (i.e. perceptual loss/feature loss) 
        or 'boundary' (weighted cross-entropy loss on the output<>semantic boundary labels).
        stimulation loss type (i.e. sparsity loss) can be either 'L1', 'L2' or None.
        """
        
        # Reconstruction loss
        if recon_loss_type == 'mse':
            self.recon_loss = torch.nn.MSELoss()
            self.target = 'image'
        elif recon_loss_type == 'vgg':
            self.feature_extractor = model.VGG_Feature_Extractor(layer_depth=recon_loss_param,device=device)
            self.recon_loss = lambda x,y: torch.nn.functional.mse_loss(self.feature_extractor(x),self.feature_extractor(y))
            self.target = 'image'
        elif recon_loss_type == 'boundary':
            loss_weights = torch.tensor([1-recon_loss_param,recon_loss_param],device=device)
            self.recon_loss = torch.nn.CrossEntropyLoss(weight=loss_weights)
            self.target = 'label'
        else:
            raise NotImplementedError

        # Stimulation loss 
        if stimu_loss_type=='L1':
            self.stimu_loss = lambda x: torch.mean(.5*(x+1)) #converts tanh to sigmoid first
        elif stimu_loss_type == 'L2':
            self.stimu_loss = lambda x: torch.mean((.5*(x+1))**2) #converts tanh to sigmoid first
        elif stimu_loss_type is None:
            self.stimu_loss = None
        self.kappa = kappa if self.stimu_loss is not None else 0
        
        # Output statistics 
        self.stats = {'tr_recon_loss':[],'val_recon_loss': [],'tr_total_loss':[],'val_total_loss':[]}
        if self.stimu_loss is not None:   
            self.stats['tr_stimu_loss']= []
            self.stats['val_stimu_loss']= []
        self.running_loss = {'recon':0,'stimu':0,'total':0}
        self.n_iterations = 0
        
        
    def __call__(self,image,label,stimulation,phosphenes,reconstruction,validation=False):    
        
        # Target
        if self.target == 'image': # Flag for reconstructing input image or target label
            target = image
        elif self.target == 'label':
            target = label
        
        # Calculate loss
        loss_stimu = self.stimu_loss(stimulation) if self.stimu_loss is not None else torch.tensor(0)
        loss_recon = self.recon_loss(reconstruction,target)
        loss_total = (1-self.kappa)*loss_recon + self.kappa*loss_stimu
        
        if not validation:
            # Save running loss and return total loss
            self.running_loss['stimu'] += loss_stimu.item()
            self.running_loss['recon'] += loss_recon.item()
            self.running_loss['total'] += loss_total.item()
            self.n_iterations += 1
            return loss_total
        else:
            # Return train loss (from running loss) and validation loss
            self.stats['val_recon_loss'].append(loss_recon.item())
            self.stats['val_total_loss'].append(loss_total.item())
            self.stats['tr_recon_loss'].append(self.running_loss['recon']/self.n_iterations)
            self.stats['tr_total_loss'].append(self.running_loss['total']/self.n_iterations)
            if self.stimu_loss is not None:
                self.stats['val_stimu_loss'].append(loss_stimu.item())
                self.stats['tr_stimu_loss'].append(self.running_loss['stimu']/self.n_iterations)  
            self.running_loss = {key:0 for key in self.running_loss}
            self.n_iterations = 0
            return self.stats


def initialize_components(cfg):
    """This function returns the required model, dataset and optimization components to initialize training.
    input: <cfg> training configuration (pandas series, or dataframe row)
    returns: dictionaries with the required model components: <models>, <datasets>,<optimization>, <train_settings>
    """

    # Random seed
    # seed = np.random.randint(1000)
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # size
    input_image_size = 128 

    # Models
    models = dict()
    models['encoder'] = NVP_v1_model.E2E_Encoder(in_channels=cfg.input_channels).to(cfg.device)
    models['decoder'] = NVP_v1_model.E2E_Decoder(out_channels=cfg.reconstruction_channels,
                                        out_activation=cfg.out_activation).to(cfg.device)
    if cfg.simulation_type == 'regular':
        pMask = utils.get_pMask(jitter_amplitude=0,dropout=False, size=(input_image_size*2,input_image_size*2)) # phosphene mask with regular mapping
    elif cfg.simulation_type == 'personalized':
        pMask = utils.get_pMask(seed=1,jitter_amplitude=.5,dropout=True,perlin_noise_scale=.4, size=(input_image_size*2,input_image_size*2)) # pers. phosphene mask
    models['simulator'] = NVP_v1_model.E2E_PhospheneSimulator(pMask=pMask.to(cfg.device),
                                                        scale_factor = 8, #use 16 when using an extra strided convolution. #default=8,
                                                        sigma=1.5,
                                                        intensity=15,
                                                        device=cfg.device, binary_stimulation=cfg.binary_stimulation).to(cfg.device)

    print(models['encoder'])

    # Dataset
    dataset = dict()
    if cfg.dataset == 'characters':
        trainset = local_datasets.Character_Dataset(device=cfg.device, imsize = (input_image_size,input_image_size))
        valset = local_datasets.Character_Dataset(device=cfg.device,validation = True, imsize = (input_image_size,input_image_size)) 
    elif cfg.dataset == 'ADE20K':
        trainset = local_datasets.ADE_Dataset(directory='/mnt/data/datasets/ade20/ADE20K',device=cfg.device)
        valset = local_datasets.ADE_Dataset(directory='/mnt/data/datasets/ade20/ADE20K',device=cfg.device,validation=True)
    dataset['trainloader'] = DataLoader(trainset,batch_size=int(cfg.batch_size),shuffle=True)
    dataset['valloader'] = DataLoader(valset,batch_size=int(cfg.batch_size),shuffle=True)

    # Optimization
    optimization = dict()
    if cfg.optimizer == 'adam':
        optimization['encoder'] = torch.optim.Adam(models['encoder'].parameters(),lr=cfg.learning_rate)
        optimization['decoder'] = torch.optim.Adam(models['decoder'].parameters(),lr=cfg.learning_rate)
    elif cfg.optimizer == 'sgd':
        optimization['encoder'] = torch.optim.SGD(models['encoder'].parameters(),lr=cfg.learning_rate)
        optimization['decoder'] = torch.optim.SGD(models['decoder'].parameters(),lr=cfg.learning_rate)
    optimization['lossfunc'] = CustomLoss(recon_loss_type=cfg.reconstruction_loss,
                                                recon_loss_param=cfg.reconstruction_loss_param,
                                                stimu_loss_type=cfg.sparsity_loss,
                                                kappa=cfg.kappa,
                                                device=cfg.device)                                   
    
    # optimization['encoder_scheduler'] = StepLR(optimization['encoder'], step_size=10, gamma=0.1)
    # optimization['decoder_scheduler'] = StepLR(optimization['decoder'], step_size=10, gamma=0.1)

    # Additional train settings
    train_settings = dict()
    if not os.path.exists(cfg.savedir):
        os.makedirs(cfg.savedir)
    train_settings['model_name'] = cfg.model_name
    train_settings['savedir']=cfg.savedir
    train_settings['n_epochs'] = cfg.n_epochs
    train_settings['log_interval'] = cfg.log_interval
    train_settings['convergence_criterion'] = cfg.convergence_crit
    return models, dataset, optimization, train_settings
    


def train(models, dataset, optimization, train_settings):
    
    ## A. Unpack parameters
   
    # Models
    encoder   = models['encoder']
    decoder   = models['decoder']
    simulator = models['simulator']
    
    # Dataset
    trainloader = dataset['trainloader']
    valloader   = dataset['valloader']
    
    # Optimization
    encoder_optim = optimization['encoder']
    decoder_optim = optimization['decoder']
    loss_function = optimization['lossfunc']

    
    # Train settings
    model_name   = train_settings['model_name']
    savedir      = train_settings['savedir']
    n_epochs     = train_settings.get('n_epochs',2)
    log_interval = train_settings.get('log_interval',10)
    converg_crit = train_settings.get('convergence_criterion',50)
    
    
    ## B. Logging
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    logger = utils.Logger(os.path.join(savedir,'out.log'))
    csvpath = os.path.join(savedir,model_name+'_train_stats.csv')
    logstats = list(loss_function.stats.keys())
    with open(csvpath, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['epoch','i']+logstats)
    logger(cfg)
    
    ## Export training/validation loader images. 
    # from torchvision.utils import save_image
    # loader_to_export = valloader
    # for i, data in enumerate(loader_to_export, 0):
    #     image,label = data
    #     ic(image.shape)
    #     for img_i in range(image.shape[0]):
    #         img = image[img_i] 
    #         # img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
    #         string = 'val_images/img_'+ str(i) + '_' + str(img_i) +'.png'
    #         save_image(img, string)
    # exit()

    ## C. Training Loop
    n_not_improved = 0
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        logger('Epoch %d' % (epoch+1))

        for i, data in enumerate(trainloader, 0):
            image,label = data

            # TRAINING
            encoder.train()
            decoder.train()
            encoder.zero_grad()
            decoder.zero_grad()

            # 1. Forward pass
            stimulation = encoder(image)
            phosphenes  = simulator(stimulation)
            reconstruction = decoder(phosphenes)

            # 2. Calculate loss
            loss = loss_function(image=image,
                                 label=label,
                                 stimulation=encoder.out,
                                 phosphenes=phosphenes,
                                 reconstruction=reconstruction)

            # 3. Backpropagation
            loss.backward()
            encoder_optim.step()
            decoder_optim.step()


            # VALIDATION
            if i==len(trainloader) or i % log_interval == (log_interval-1):
                image,label = next(iter(valloader))

                encoder.eval()
                decoder.eval()

                # if(cfg.NVP_v1_QAT):
                #     # Convert the observed model to a quantized model. This does several things:
                #     # quantizes the weights, computes and stores the scale and bias value to be
                #     # used with each activation tensor, fuses modules where appropriate,
                #     # and replaces key operators with quantized implementations.
                #     # -> relevant calculations will happen in int8
                #       encoder_int8 = torch.quantization.convert(encoder)

                with torch.no_grad():
                    stimulation = encoder(image)
                    phosphenes  = simulator(stimulation)
                    reconstruction = decoder(phosphenes)            

                    # 2. Loss         
                    stats = loss_function(image=image,
                                            label=label,
                                            stimulation=encoder.out,
                                            phosphenes=phosphenes,
                                            reconstruction=reconstruction,
                                            validation=True)            
                               
                # 3. Logging
                logstats = ' | '.join('%s : %.3f' %(key,stats[key][-1]) for key in stats) 
                logger('[%d, %5d] %s' %(epoch,i + 1, logstats))
                # logger(optimization['encoder_scheduler'].get_lr(), optimization['decoder_scheduler'])
                with open(csvpath, 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([epoch,i + 1]+[stats[key][-1] for key in stats])                

                # # 4. Visualization
                # plt.figure(figsize=(10,10),dpi=50)
                # utils.plot_stats(stats)
                # plt.figure(figsize=(10,10),dpi=50)
                # utils.plot_images(image[:5])
                # plt.figure(figsize=(10,10),dpi=50)
                # utils.plot_images(phosphenes[:5])
                # plt.figure(figsize=(10,10),dpi=50)
                # utils.plot_images(reconstruction[:5])
                # if len(label.shape)>1:
                #     plt.figure(figsize=(10,10),dpi=50)
                #     utils.plot_images(label[:5])    

                # # Calculate performance metrics
                # im_pairs = [[im.squeeze().cpu().numpy(),trg.squeeze().cpu().numpy()] for im,trg in zip(image,reconstruction)]
                
                # if cfg.reconstruction_loss == 'boundary':
                #     metrics=pd.Series() #TODO
                # else:
                #     mse = [mean_squared_error(*pair) for pair in im_pairs]
                #     ssim = [structural_similarity(*pair, gaussian_weigths=True) for pair in im_pairs]
                #     psnr = [peak_signal_noise_ratio(*pair) for pair in im_pairs]
                #     metrics=pd.Series({'mse':np.mean(mse),
                #                     'ssim':np.mean(ssim),
                #                     'psnr':np.mean(psnr)})
                #     print(sum(mse) / len(mse))
                
                # 5. Save model (if best)
                # ic(len(stats['val_total_loss']))
                # ic(np.argmin(stats['val_total_loss'])+1)
                if  np.argmin(stats['val_total_loss'])+1==len(stats['val_total_loss']):
                    savepath = os.path.join(savedir,model_name + '_best_encoder.pth' )#'_e%d_encoder.pth' %(epoch))#,i))
                    logger('Saving to ' + savepath + '...')
                    torch.save(encoder.state_dict(), savepath)

                    savepath = os.path.join(savedir,model_name + '_best_decoder.pth' )#'_e%d_decoder.pth' %(epoch))#,i))
                    logger('Saving to ' + savepath + '...')
                    torch.save(decoder.state_dict(), savepath)
                    
                    n_not_improved = 0
                else:
                    n_not_improved = n_not_improved + 1
                    logger('not improved for %5d iterations' % n_not_improved) 
                    if n_not_improved>converg_crit:
                        break

                # 5. Prepare for next iteration
                encoder.train()
                decoder.train()            


            # optimization['encoder_scheduler'].step()
            # optimization['decoder_scheduler'].step()

        if n_not_improved>converg_crit:
            break
    logger('Finished Training')
    
    return {'encoder': encoder, 'decoder':decoder}, loss_function.stats

if __name__ == '__main__':
    import argparse
    import pandas as pd
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_name", type=str, default="demo_model",
                    help="model name")
    ap.add_argument("-dir", "--savedir", type=str, default="nvp_NN_ade20k_personalized_vgg/",
                    help="directory for saving the model parameters and training statistics")
    ap.add_argument("-s", "--seed", type=int, default=0,
                    help="seed for random initialization")
    ap.add_argument("-e", "--n_epochs", type=int, default=80, #80
                    help="number of training epochs")   
    ap.add_argument("-l", "--log_interval", type=int, default=50,
                    help="number of batches after which to evaluate model (and logged)")   
    ap.add_argument("-crit", "--convergence_crit", type=int, default=100,
                    help="stop-criterion for convergence: number of evaluations after which model is not improved")   
    ap.add_argument("-bin", "--binary_stimulation", type=bool, default=True,
                    help="use quantized (binary) instead of continuous stimulation protocol")   
    ap.add_argument("-sim", "--simulation_type", type=str, default="personalized",
                    help="'regular' or 'personalized' phosphene mapping") 
    ap.add_argument("-in", "--input_channels", type=int, default=1,
                    help="only grayscale (single channel) images are supported for now")   
    ap.add_argument("-out", "--reconstruction_channels", type=int, default=1,
                    help="only grayscale (single channel) images are supported for now")     
    ap.add_argument("-act", "--out_activation", type=str, default="sigmoid",
                    help="use 'sigmoid' for grayscale reconstructions, 'softmax' for boundary segmentation task")   
    ap.add_argument("-d", "--dataset", type=str, default="ADE20K",
                    help="'charaters' dataset and 'ADE20K' are supported")   
    ap.add_argument("-dev", "--device", type=str, default="cuda:0",
                    help="e.g. use 'cpu' or 'cuda:0' ")   
    ap.add_argument("-n", "--batch_size", type=int, default=32,
                    help="'charaters' dataset and 'ADE20K' are supported")   
    ap.add_argument("-opt", "--optimizer", type=str, default="adam",
                    help="only 'adam' is supporte for now")   
    ap.add_argument("-lr", "--learning_rate", type=float, default=0.0002,
                    help="Use higher learning rates for VGG-loss (perceptual reconstruction task)")  
    ap.add_argument("-rl", "--reconstruction_loss", type=str, default='vgg',
                    help="'mse', 'vgg' or 'boundary' loss are supported ") 
    ap.add_argument("-p", "--reconstruction_loss_param", type=float, default=2.0,
                    help="In perceptual condition: the VGG layer depth, boundary segmentation: cross-entropy class weight") 
    ap.add_argument("-L", "--sparsity_loss", type=str, default='L1',
                    help="choose L1 or L2 type of sparsity loss (MSE or L1('taxidrivers') norm)") 
    ap.add_argument("-k", "--kappa", type=float, default=0.03,
                    help="sparsity weight parameter kappa")
    cfg = pd.Series(vars(ap.parse_args()))

    print(cfg)
    models, dataset, optimization, train_settings = initialize_components(cfg)

    # # Print MACs and model size (using https://github.com/Lyken17/pytorch-OpCounter)
    # from thop import profile
    # input = torch.randn(1, 1, 64, 32).cuda()
    # macs, params = profile(models["encoder"], inputs=(input, ))
    # print("macs, params")
    # print(macs, params)
    # exit()

    path = os.path.abspath(os.getcwd())

    shutil.copy("NVP_v1_model.py", cfg.savedir+"NVP_v1_model.py")

    # run the training loop
    train(models, dataset, optimization, train_settings)



    

