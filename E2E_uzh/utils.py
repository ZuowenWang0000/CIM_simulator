import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
import datetime
import logging
import torchvision
import noise
import pandas as pd
import os 
import model


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


class Logger(object):
    def __init__(self,log_file='out.log'):
        self.logger = logging.getLogger()
        hdlr = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr) 
        self.logger.setLevel(logging.INFO)
    def __call__(self,message):
        #outputs to Jupyter console
        print('{} {}'.format(datetime.datetime.now(), message))
        #outputs to file
        self.logger.info(message)

        
        
def get_pMask(size=(256,256),phosphene_density=32,seed=1,jitter_amplitude=0,dropout=False,perlin_noise_scale=.4):

    # Define resolution and phosphene_density
    # print(size)
    [nx,ny] = size
    n_phosphenes = phosphene_density**2 # e.g. n_phosphenes = 32 x 32 = 1024
    pMask = torch.zeros(size)


    # Custom 'dropout_map'
    p_dropout = perlin_noise_map(shape=size,scale=perlin_noise_scale*size[0],seed=seed)
    np.random.seed(seed)

    for p in range(n_phosphenes):
        i, j = divmod(p, phosphene_density)

        jitter = np.round(np.multiply(np.array([nx,ny])//phosphene_density, jitter_amplitude * (np.random.rand(2)-.5))).astype(int)
        rx = (j*nx//phosphene_density) + nx//(2*phosphene_density) + jitter[0]
        ry = (i*ny//phosphene_density) + ny//(2*phosphene_density) + jitter[1]

        rx = np.clip(rx,0,nx-1)
        ry = np.clip(ry,0,ny-1)
        if dropout==True:
            pMask[rx,ry] = np.random.choice([0.,1.], p=[p_dropout[rx,ry],1-p_dropout[rx,ry]])
        else:
            pMask[rx,ry] = 1.
            
    return pMask       
 

def perlin_noise_map(seed=0,shape=(256,256),scale=100,octaves=6,persistence=.5,lacunarity=2.):
    out = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            out[i][j] = noise.pnoise2(i/scale, 
                                        j/scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=shape[0], 
                                        repeaty=shape[1], 
                                        base=seed)
    out = (out-out.min())/(out.max()-out.min())
    return out

def plot_stats(stats):
    """ Plot dict containing lists of train statistics"""
    for key in stats:
        plt.plot(stats[key], label=key)
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('training statistics')
    plt.show()
    return


# For basic plotting of images with labels as title
def plot_images(img_tensor,title=None,classes=None):
    
    # Un-normalize if images are normalized  
    if img_tensor.min()<0:
        if img_tensor.shape[1]==3:
            normalizer = TensorNormalizer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif img_tensor.shape[1]==1:
            normalizer = TensorNormalizer(mean=0.459, std=0.227)
        img_tensor = normalizer.undo(img_tensor)
        
    # Make numpy
    img = img_tensor.detach().cpu().numpy()  
    
    
    # Plot all
    for i in range(len(img)):    
        plt.subplot(1,len(img),i+1)
        if type(title) is list:
            plt.title(title[i])
        elif title is not None and classes is not None:
            plt.title(classes[title[i].item()])
        if img.shape[1]==1 or len(img.shape)==3:
            plt.imshow(np.squeeze(img[i]),cmap='gray',vmin=0,vmax=1)
        elif img.shape[1]==2:    
            plt.imshow(img[i][1],cmap='gray',vmin=0,vmax=1)
        else:
            plt.imshow(img[i].transpose(1,2,0))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    return


# To do (or undo) normalization on torch tensors
class TensorNormalizer(object):
    """To normalize and un-normalize image tensors. For grayscale images uses scalar values for mean and std.
    When called, the  number of channels is automatically inferred."""
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std  = std
    def __call__(self,image):
        if image.shape[1]==3:
            return torch.stack([(image[:, c, :, :] - self.mean[c]) / self.std[c] for c in range(3)],dim=1)
        else:
            return (image-self.mean)/self.std
    def undo(self,image):
        if image.shape[1]==3:
            return torch.stack([image[:, c, :, :]* self.std[c] + self.mean[c] for c in range(3)],dim=1)
        else:
            return image*self.std+self.mean

def add_noise(clean_image, level=0.3):
    """Inverts random elements of the original image
    value of noise level should be chosen in range [0. , 0.5]"""
    # Add noise (random inversion of the image)
    image = clean_image.clone()
    mask = np.random.randint(0,image.numel(),int(level*image.numel()))
    image.flatten()[mask] = 1-image.flatten()[mask]
    return image

    
    
# To convert to 3-channel format (or reversed)
class RGBConverter(object):
    def __init__(self,weights=[.3,.59,.11]):
        self.weights=weights
        self.copy_channels = torchvision.transforms.Lambda(lambda img:img.repeat(1,3,1,1))
    def __call__(self,image):
        assert len(image.shape) == 4 and image.shape[1] == 1
        image = self.copy_channels(image)
        return image
    def to_gray(self,image):
        assert len(image.shape) == 4 and image.shape[1] == 3
        image = torch.stack([self.weights[c]*image[:,c,:,:] for c in range(3)], dim=1)
        image = torch.sum(image,dim=1,keepdim=True)
        return image
    