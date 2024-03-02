import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Lambda
from PIL import Image
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm import tqdm
from math import ceil, floor

class Diffusion:
    """
    The Diffuser class, the main class.
    """
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, channels=3, device='cuda'):
        """
        Initialisation. This sets the most important parameters.
        args:
            - noise_steps (int): The number of steps between the original image and complete noise.
                                 The original paper uses 1000 steps. 
                                 However, more is better for image quality (will yield longer sampling times as well).
            - beta_start (float): The start of the noise schedule
            - beta_end (float): The end of the noise schedule
            - img_size (int): The image size (assumes rectangular images).
            - channels (int): The number of channels in the input images
            - device (str): The device used for training
        """
        self.noise_steps = noise_steps 
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.channels = channels
        self.device = device        
        
        self.beta = self.prepare_noise_schedule().to(device) #Create the noise schedule based on beta_start and beta_end
        self.alpha = 1. - self.beta #Create the alphas needed for adding noise to the images
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) #Create cumulative alphas for adding noise to the images (also called alpha_hat)
        
    def prepare_noise_schedule(self):
        """
        This function creates the noise schedule based on beta_start and beta_end.
        More specifically, this creates a linear noise schedule (at each time step the same amoutnof noise is added on top of the previous noise).
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        """
        This takes a batch of images and adds noise to them based on the timestep.
        This uses the formula xt = sqrt(alpha_hat of t)*x + sqrt(1-alpha_hat of t)*e with x the original images and e the noise
        args:
            - x (float, normalised image with values between [-1,1]): A batch of original images.
            - t (int): A batch of timesteps
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] #Create the sqrt and expand the dimensions
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None] #Create the sqrt and expand the dimensions
        e = torch.randn_like(x)
        return sqrt_alpha_hat*x + sqrt_one_minus_alpha_hat*e, e
    
    def sample_timesteps(self, n):
        """
        Create a batch of randomly generated timesteps.
        args:
            - n (int): batch size
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, model, n, verbose=0):
        """
        This function follows Algorithm 2 of the DDPM paper by Ho et al. (2020).
        Algorithm 2:
        1: sample Xt (basically noise) from a gaussain distribution with mean 0 and st.dev of 1
        2: for each timestep (from noise_steps to 0):
            sample z from a gaussain distribution with mean 0 and st.dev of 1 if t >1 else 0
            xt-1 = 1/sqrt(alpha of t) * (xt - (1 - alpha of t)/sqrt(1 - alpha_hat of t)*model(xt, t)) + sqrt(beta of t) * z
        3: return x0
        
        args:
            - model (pytorch model): The specific neural net trained for diffusion (normally this is a U-Net).
            - n (int): The batch size
            - verbose (int): Determines the amount of information given during training with 0 being the least amount of information. 

        """
        
        model.eval() #We don't need gradients and all the layers should be in eval mode/
        with torch.no_grad():
            # Create x of T which is basically random noise with the dimensions of a square image and several channels
            x = torch.randn((n, self.channels, self.img_size, self.img_size)).to(self.device) 
            
            # Denoise the samples
            for i in tqdm(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, disable=(verbose<1)): #Go from T to 0 and denoise the samples
                # Set the current timestep
                t = (torch.ones(n)*i).long().to(self.device) 
                
                # Get the prediction of the noise
                predicted_noise = model(x, t)
                
                # Calculate the parameters for Algorithm 2
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1/torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
        # Unnormalize the image from [-1, 1] to [0, 255]
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


class SelfAttention(nn.Module):
    """
    Self-attention is often used in natural language processing and computer vision. 
    The self-attention module is used to identify and weigh the importance of data by attending to itself. 
    Basically, self-attention uses the data in a sample to highlight the parts that are important for the predictions.
    """
    def __init__(self, channels, n_heads = 4):
        """
        Initializes the self_attention module.
        args:
            - channels (int): The number of channels in the input
            - n_heads (int): The number of heads of multi-head attention
        """
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        
        self.mha = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]), 
            nn.Linear(channels, channels), 
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    
    def forward(self, x):
        """
        Forward method to perform self-attention on an image
        args:
            - x (image): The input image.
        """
        size = x.shape[-1]#size (int): The image dimension (we assume square images)
        x = x.view(-1, self.channels, size*size).transpose(1, 2) #merge the spatial dimensions and get the channels at the end, basically the channels are used as 'embeddings' and the pixels as tokens
        x_ln = self.ln(x) #Normalize the pixels over the different channels
        attention_value, _ = self.mha(x_ln, x_ln, x_ln) #Perform multi-head attention on the normalized images, with channels as the ambedding
        attention_value = attention_value + x #Residual connection
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.transpose(1,2).view(-1, self.channels, size, size) #Reset the dimensions by making the channels the second one and splitting the spatial dimensions

    
class DoubleConv(nn.Module):
    """
    Combines two convolution operations and a residual connection with group normalization.
    In fact, each convolution operation is followed by a group normalization layer and a GeLU operation is performed
    after the group normalization of the first convolutional layer and the second convolutional layer.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        """
        Initializes the DoubleConv module.
        args:
            - in_channels (int): Number of channels in the input
            - out_channels (int): Number of channels in the output
            - mid_channels (int): The number of channels after the first and before the secod convolutional layer. If None, it is equal to out_channels.
            - residual (bool): If True enable the residual connection
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = out_channels
        if mid_channels is not None:
            self.mid_channels = mid_channels
        self.residual = residual
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=self.mid_channels), #Normalize all the pixels over the channels
            nn.GELU(),
            nn.Conv2d(in_channels=self.mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=out_channels),
        )
    
    def forward(self, x):
        """
        Forward method for the double convolution module.
        args:
            - x (image): The input image.
        """
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """
    Downscales and performs two DoubleConv layers, adds the embedded time information to the downscaled images.
    The spatial dimensions shrink by a factor of 2 after this layer.
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        """
        Initializes the Down module.
        args:
            - in_channels (int): Number of channels in the input
            - out_channels (int): Number of channels in the output
            - emb_dim (int): The embedding dimension for the time
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_dim = emb_dim
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, residual=False)
        )
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels) #Make the embedded time compatible with the downscaled image
        )
        
    def forward(self, x, t):
        """
        Forward method for the downscaling module.
        args:
            - x (image): The input image.
            - t (float): Embedded time.
        """
        
        x = self.maxpool_conv(x) #Downscale and perform convolution
        emb = self.emb_layer(t)[:,:, None, None] #change embedding dimension and add additional dimensions
        return x + emb #add embedding to the downsampled image


class Up(nn.Module):
    """
    Upscales, adds the corresponding image from the Down module, performs two DoubleConv layers
    and adds the embedded time information to the resulting images.
    The spatial dimensions grow by a factor of 2 after this layer.
    """
    def __init__(self, in_channels, out_channels, emb_dim=256):
        """
        Initializes the Up module.
        args:
            - in_channels (int): Number of channels in the input
            - out_channels (int): Number of channels in the output
            - emb_dim (int): The embedding dimension for the time
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_dim = emb_dim
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels=in_channels, out_channels=in_channels, residual=True),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels//2, residual=False)
        )
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels) #Make the embedded time compatible with the upscaled image
        )
        
    def forward(self, x, skip_x, t):
        """
        Forward method for the downscaling module.
        args:
            - x (image): The input image.
            - skip_x (image): The image from the corresponding Down module)
            - t (float tensor): Embedded time.
        """
        
        x = self.up(x) #Upscale the input
        x = torch.cat([skip_x, x], dim=1) #Add the image from the corresponding Down Module to the upscaled image
        x = self.conv(x) #Perform convolution
        emb = self.emb_layer(t)[:,:, None, None] #change embedding dimension and add additional dimensions
        return x + emb #add embedding to the downsampled image


class UNet(nn.Module):
    """
    U-Net is a popular convolutional neural network originally created for image segmentation. 
    This model combines downsampling by pooling to contract the spatial image dimensions 
    followed by upsampling to recreate the original dimensions (U-shape). The usage of a large number
    of channels allows for the proper propagation of information. Basically the input of the model 
    is an image and the output is an image (for example segmented image). The U-net consiste of an encoder (downsampling)
    and a decoder (upsampling) with a bottleneck in between where the similar encoder and decoder parts are 
    connected, creating a secondary information flow from the encoder to the decoder (next to the flow through the 'U' of the U-Net).
    
    The U-Net used by the authors of the DDPM paper by Ho et al. (2020) is actually an advanced form of a U-Net where they
    added an amalgamation of a lot of existing improvements. 
    """
    def __init__(self, in_channels=3, out_channels=3, time_dim=256, base_channels=64, n_scaling=3, n_bottleneck=3, device='cuda'):
        """
        Initializes the U-Net.
        args:
            - in_channels (int): The number of channels of the input
            - out_channels (int): The number of channels of the output
            - time_dim (int): The dimension of the time embedding.
            - base_channels (int): Number of channels for the first and last scaling layer (will double until the middle most layer)
            - n_scaling (int): Number of scaling layers
            - n_bottleneck (int): Number of bottleneck layers
            - device (str): The device used for training
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.base_channels = base_channels
        self.n_scaling = n_scaling
        self.n_bottleneck = n_bottleneck
        self.device = device
               
        self.down_block = nn.ModuleList([])
        self.down_att = nn.ModuleList([])
        self.up_block = nn.ModuleList([])
        self.up_att = nn.ModuleList([])
        self.bottleneck_block = nn.ModuleList([])
        
        self.inc = DoubleConv(in_channels=in_channels, out_channels=base_channels)
        
        for i in range(0, n_scaling-1):
            self.down_block.append(Down(in_channels=base_channels*2**i, out_channels=base_channels*2**(i+1)))
            self.down_att.append(SelfAttention(channels=base_channels*2**(i+1)))     
        
        self.down_block.append(Down(in_channels=base_channels*2**(n_scaling-1), out_channels=base_channels*2**(n_scaling-1)))
        self.down_att.append(SelfAttention(channels=base_channels*2**(n_scaling-1)))
        
        for i in range(0, n_bottleneck//2):
            self.bottleneck_block.append(DoubleConv(in_channels=base_channels*2**(n_scaling - 1 + i), out_channels=base_channels*2**(n_scaling - 1 + i + 1)))
        
        if n_bottleneck%2 != 0:
            self.bottleneck_block.append(DoubleConv(in_channels=base_channels*2**(n_scaling - 1 + n_bottleneck//2), out_channels=base_channels*2**(n_scaling - 1 + n_bottleneck//2)))
        
        for i in range(n_bottleneck//2, 0, -1):
            self.bottleneck_block.append(DoubleConv(in_channels=base_channels*2**(n_scaling - 1 + i), out_channels=base_channels*2**(n_scaling - 1 + i - 1)))
        
        for i in range(n_scaling-1, 0, -1):
            self.up_block.append(Up(in_channels=base_channels*2**(i+1), out_channels=base_channels*2**(i-1))) # the up_block will be half from the previous block and half from the corresponding down block
            self.up_att.append(SelfAttention(channels=int(base_channels*2**(i-1))))

        self.up_block.append(Up(in_channels=base_channels*2, out_channels=base_channels)) # the up_block will be half from the previous block and half from the corresponding down block
        self.up_att.append(SelfAttention(channels=int(base_channels)))

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)    
    
    def pos_encoding(self, t):
        """
        Implements the positional encoding for the time dimension
        args:
            - t (float): A batch of time.
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.time_dim, 2, device=self.device).float() / self.time_dim))
        pos_enc_a = torch.sin(t.repeat(1, self.time_dim//2)*inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, self.time_dim//2)*inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
        
    def forward(self, x, t):
        """
        Forward method for the U-Net module.
        args:
            - x (image): A batch of noisy images.
            - t (int): A batch of time.
        """
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t)
        
        down_outs = []
        x = self.inc(x)
        down_outs.append(x)
        
        for i in range(self.n_scaling):
            x = self.down_block[i](x, t)
            x = self.down_att[i](x)
            down_outs.append(x)
            
        for i in range(self.n_bottleneck):
            x = self.bottleneck_block[i](x)
        
        down_outs.pop()
        for i in range(self.n_scaling):
            x = self.up_block[i](x, down_outs.pop(), t)
            x = self.up_att[i](x)
        
        output = self.outc(x)
        return output


def generate_image_batch(model, diffusion, n_images, batch_size=None, verbose=0):
    """
    Function to generate a large batch of images.
    args:
        - model (nn.Module): The model (most likely a U-Net) trained to predict the noise in an image.
        - diffusion (instance of the Diffusion class): diffusion instance to add noise to images
        - n_images (int): The number of images to be generated.
        - batch_size (int): The batch size for generation.
        - verbose (int): Determines the amount of information given during training with 0 being the least amount of information. 
    """
    if batch_size is None:
        batch_size = n_images
    
    batches = ceil(n_images/batch_size)
    
    images = []
    
    for i in tqdm(range(batches), total=batches, disable=(verbose<1)):
        images.append(diffusion.sample(model, batch_size).cpu())
    
    return torch.cat(images, dim=0)


def plot_images(images, figsize=(10,10), columns=5, color=False):
    """
    Plots generated images.
    args:
        - images (torch.tensor): Batch of generated images.
        - figsize (tuple): The size of the figure.
        - columns (int): The number of images per row
        - color (bool): Whether the image is in color
    """
    images = images.clone()
    length = images.shape[0]
    
    rows = ceil(length/columns)
    fig, axs = plt.subplots(rows, columns, figsize=figsize)
    axs = axs.flatten()
    
    for i, ax in enumerate(axs):
        if i < length:
            if color:
                ax.imshow(images[i].permute(1,2,0))
            else:
                ax.imshow(images[i][0])
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    

def train(dataloader, model, optimizer, diffusion, epochs, device='cuda', target=True, verbose=0,
          validation=None, img_args=None):
    """
    Training function to train a model to predict the noise in images.
    args:
        - dataloader (torch.utils.data.DataLoader): Dataloader for the training images
        - model (torch.nn.Module): The model (most likely a U-Net) that will be trained to predict the noise in an image
        - optimizer (torch.optim optimizer): The optimizer to update the model.
        - diffusion (instance of the Diffusion class): diffusion instance to add noise to images
        - epochs (int): Number of epochs to train for.
        - device (str): The device for training.
        - target (bool): Whether the target is present in the dataloader or not. 
        - verbose (int): Determines the amount of information given during training with 0 being the least amount of information. 
        - validation (int): Number of images to be sampled from the diffusion process as validation. If None, no images will be taken
        - img_args (dict): Parameters for imaging the validation images with the function plot_images.
    """
    validation_result = []
    
    mse = nn.MSELoss()
    model.to(device)
    length = len(dataloader)
    
    model.train()
    for epoch in range(epochs):
        print(f"Start training {epoch+1}/{epochs}")
        train_loss = 0
        if verbose > 0:
              
            start = timer()
        for i, sample in tqdm(enumerate(dataloader), total=len(dataloader), disable=(verbose<1)):
            if target:
                images, _ = sample
            else:
                images = sample
            
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().detach()
        
        if verbose > 0:
            end = timer()
            current_time = end - start
            minutes = int(current_time/60)
            seconds = current_time - minutes*60

            print(f"Epoch {epoch+1}/{epochs} - training loss: {train_loss/len(dataloader):.4f} - {minutes}m {seconds:2f}s")
        
        if validation is not None:
            print(f"Start validation")
            new_img = diffusion.sample(model, validation, verbose).cpu()
            validation_result.append(new_img)
            if img_args is not None:
                plot_images(new_img, **img_args)
            model.train()
            
    if validation is not None:
        return validation_result
