from simple_diffuser_improved_cosine import *
import numpy as np

class Diffusion:
    """
    The Diffuser class, the main class.
    """
    def __init__(self, noise_steps=1000, noise_schedule="linear", beta_start=1e-4, beta_end=0.02, s=0.008, scale_time=True, img_size=64, channels=3, device='cuda'):
        """
        Initialisation. This sets the most important parameters.
        args:
            - noise_steps (int): The number of steps between the original image and complete noise.
                                 The original paper uses 1000 steps. 
                                 However, more is better for image quality (will yield longer sampling times as well).
            - noise_schedule (str): Is either "linear", "quadratic", "sigmoid" or "cosine". The linear schedule works very well for images with high resolution (due to high increase of noise). 
                                    Cosine is better for smaller images (the noise addition is more gradual). 
            - beta_start (float): The start of the noise schedule
            - beta_end (float): The end of the noise schedule
            - s (float): The cosine schedule offset
            - scale_time (bool): Whether to scale the betas with time or not (one of the improvements from Nichol & Dhariwal (2021))
            - img_size (int): The image size (assumes rectangular images).
            - channels (int): The number of channels in the input images
            - device (str): The device used for training
        """
        self.noise_steps = noise_steps 
        self.noise_schedule = noise_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.s = s
        self.scale_time = scale_time
        self.img_size = img_size
        self.channels = channels
        self.device = device        
        
        self.beta = self.get_noise_schedule().to(device) #Create the noise schedule based
        self.alpha = 1. - self.beta #Create the alphas needed for adding noise to the images
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) #Create cumulative alphas for adding noise to the images (also called alpha_hat)
        
    def get_noise_schedule(self):
        if self.noise_schedule == 'linear':
            return self.prepare_linear_schedule()
        elif self.noise_schedule == 'cosine':
            return self.prepare_cosine_schedule()
        elif self.noise_schedule == 'quadratic':
            return self.prepare_quadratic_schedule()
        elif self.noise_schedule == 'sigmoid':
            return self.prepare_sigmoid_schedule()
        raise  AssertionError('Not a valid schedule')

    def prepare_linear_schedule(self):
        """
        This function creates the improved linear noise schedule based on Improved denoising diffusion probabilistic models from Nichol & Dhariwal (2021).
        At each time step the same amount of noise is added on top of the previous noise.
        """
        scale = 1.0
        if self.scale_time:
            scale = 1000/self.noise_steps
        beta_start = scale * self.beta_start
        beta_end = scale * self.beta_end
        return torch.linspace(beta_start, beta_end, self.noise_steps)
    
    def prepare_cosine_schedule(self):
        """
        This function creates the cosine noise schedule based on Improved denoising diffusion probabilistic models from Nichol & Dhariwal (2021) and the annotated diffusion model from hugging face.
        The addition of noise is more smooth compared to the linear schedule, making the later timepoints (with a lot of noise) more informative.
        """        
        steps = self.noise_steps + 1
        t = torch.linspace(0, self.noise_steps, steps)
        alphas_cumprod = torch.cos(((t / self.noise_steps) + self.s) / (1 + self.s) * torch.pi/2.0) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        scale = 1.0
        if self.scale_time:
            scale = 1000/self.noise_steps
        beta_start = scale * self.beta_start
        beta_end = scale * self.beta_end
        return torch.clip(betas, self.beta_start, self.beta_end)
    
    def prepare_quadratic_schedule(self):
        r"""
        This function creates the quadratic noise schedule based on Improved denoising diffusion probabilistic models from Nichol & Dhariwal (2021) and the annotated diffusion model from hugging face.
        """
        scale = 1.0
        if self.scale_time:
            scale = 1000/self.noise_steps
        beta_start = scale * self.beta_start
        beta_end = scale * self.beta_end
        return torch.linspace(beta_start**0.5, beta_end**0.5, self.noise_steps)**2

    def prepare_sigmoid_schedule(self):
        r"""
        This function creates the quadratic noise schedule based on Improved denoising diffusion probabilistic models from Nichol & Dhariwal (2021) and the annotated diffusion model from hugging face.
        """
        scale = 1.0
        if self.scale_time:
            scale = 1000/self.noise_steps
        beta_start = scale * self.beta_start
        beta_end = scale * self.beta_end
        betas = torch.linspace(-6, 6, self.noise_steps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start    
    
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
        
        #print(t[0])
        #plt.imshow((sqrt_alpha_hat*x + sqrt_one_minus_alpha_hat*e).cpu()[0][0])
        #plt.show()
        return sqrt_alpha_hat*x + sqrt_one_minus_alpha_hat*e, e
    
    def sample_timesteps(self, n):
        """
        Create a batch of randomly generated timesteps.
        args:
            - n (int): batch size
        """
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, model, n, labels=None, cfg_scale=3, verbose=0):
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
            - labels (tensor): The labels for the images, if None, images will be samples unconditionally
            - cfg_scale (float): The guidance scale for image generation, a higher value means more guidance.
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
                
                
                # Get the prediction of the noise (unconditional)
                predicted_noise = model(x, t)
                
                if labels is not None and cfg_scale>0:
                    # Get the prediction of the noise (conditional)
                    predicted_noise_c = model(x, t, labels)
                    
                    # Create the overall predicted noise
                    predicted_noise = torch.lerp(predicted_noise, predicted_noise_c, cfg_scale)
                
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
    def __init__(self, in_channels=3, out_channels=3, time_dim=256, num_classes=None, base_channels=64, n_scaling=3, n_bottleneck=3, device='cuda'):
        """
        Initializes the U-Net.
        args:
            - in_channels (int): The number of channels of the input
            - out_channels (int): The number of channels of the output
            - time_dim (int): The dimension of the time embedding.
            - num_classes (int): The number of classes for conditional diffusion, if unconditional diffusion is needed use None as input.
            - base_channels (int): Number of channels for the first and last scaling layer (will double until the middle most layer)
            - n_scaling (int): Number of scaling layers
            - n_bottleneck (int): Number of bottleneck layers
            - device (str): The device used for training
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.num_classes = num_classes
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
        
        if self.num_classes is not None:
            self.label_embed = nn.Embedding(num_classes, time_dim) # If conditional diffusion is needed => the embedding dimension of the classes should be the same as the time_dim so we can add it to the embedded time
    
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
        
    def forward(self, x, t, y=None):
        """
        Forward method for the U-Net module.
        args:
            - x (image): A batch of noisy images.
            - t (int): A batch of time.
        """
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t)
        
        if y is not None:
            t += self.label_embed(y)
        
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


def train(dataloader, model, optimizer, diffusion, epochs, device='cuda', conditional=True, target=True, verbose=0,
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
        - conditional (bool): If conditional diffusion is needed set to True, labels are expacted to be present in the dataloader. 
        - target (bool): Whether the target is present in the dataloader or not (ignored when conditional diffusion is set to True). 
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
            
            if conditional:
                images, labels = sample      
                images, labels = images.to(device), labels.to(device)
            else:
                if target:
                    images, _ = sample
                else:
                    images = sample            
                images = images.to(device)
            
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            
            if conditional:
                if np.random.random() < 0.1:
                    labels = None                
                predicted_noise = model(x_t, t, labels)            
            else:
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
            new_img = diffusion.sample(model, validation, verbose=verbose).cpu()
            validation_result.append(new_img)
            if img_args is not None:
                plot_images(new_img, **img_args)
            model.train()
            
    if validation is not None:
        return validation_result
