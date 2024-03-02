from simple_diffuser_improved import *

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
