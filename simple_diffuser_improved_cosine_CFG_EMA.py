from simple_diffuser_improved_cosine_CFG import *
from copy import deepcopy

class EMA:
    """
    Exponential moving average class. Exponential moving average works by making small conservative updates to a central model. 
    In a normal machine learning model training program, the parameters would be directly updated. Here the average of the old and new parameters 
    are used to update the model. In many cases, having a conservative update can make training more smooth. However, in some cases the noise from 
    gradient updates might be beneficial as it has a regularizing effect.
    
    EMA works like this: updated_weights = old_weights * beta + (1 - beta) * new_weights
    """
    
    def __init__(self, beta, warmup=2000):
        """
        Initialisation.
        args:
            - beta (floart): This parameter determines how much of the old model should be retained in gradient updates. 
            - warmup (int): The number of warming up steps needed. During warming up, EMA will not work and the new weights will be returned.
        """
        
        self.beta = beta
        self.warmup = warmup
        self.step = 0              
        
    def __call__(self, ema_model, new_model):
        """
        The function to call the class like a function.
        This takes the new updated model and uses it to update the EMA model. 
        During warmup the EMA model is just a copy of the new model.
        args:
            - ema_model (nn.Module): The EMA pytorch model
            - new_model (nn.Module): The pytorch model that is used for training
        """
        # If the warmuop is taking place, return the new model as the EMA model
        if self.step < self.warmup:
            ema_model.load_state_dict(new_model.state_dict())
            self.step += 1
            return
        
        # Update all parameters of the EMA model
        for ema_params, new_params in zip(ema_model.parameters(), new_model.parameters()):
            ema_weight, new_weight = ema_params.data, new_params.data
            if ema_weight is None:
                ema_params.data = new_weight
            else:
                ema_params.data = ema_weight*self.beta + (1 - self.beta)*new_weight
        return
      
      
def train(dataloader, model, optimizer, diffusion, epochs, device='cuda', conditional=True, target=True, verbose=0,
          validation=None, img_args=None, EMA_beta=None, EMA_warmup=None):
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
        - EMA_beta (float): If no EMA is needed set to None. Otherwise, this expects a float between 0 or 1 showing how conservative the EMA updates should be.
        - EMA_warmup (int): If no EMA is needed set to None. Otherwise, this is the number of warmup steps for EMA. 
    """
    validation_result = []
    
    mse = nn.MSELoss()
    model.to(device)
    length = len(dataloader)
    
    if EMA_beta is not None and EMA_warmup is not None:
        ema = EMA(EMA_beta, EMA_warmup)
        ema_model = deepcopy(model).eval().requires_grad_(False)
    
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
            
            if EMA_beta is not None and EMA_warmup is not None: 
                ema(ema_model, model)
        
        if verbose > 0:
            end = timer()
            current_time = end - start
            minutes = int(current_time/60)
            seconds = current_time - minutes*60

            print(f"Epoch {epoch+1}/{epochs} - training loss: {train_loss/len(dataloader):.4f} - {minutes}m {seconds:2f}s")
        
        if validation is not None:
            print(f"Start validation")
            if EMA_beta is not None and EMA_warmup is not None:
                new_img = diffusion.sample(ema_model, validation, verbose=verbose).cpu()
            else:
                new_img = diffusion.sample(model, validation, verbose=verbose).cpu()
                model.train()
            validation_result.append(new_img)
            if img_args is not None:
                plot_images(new_img, **img_args)            
    
    if EMA_beta is not None and EMA_warmup is not None: 
        model.load_state_dict(ema_model.state_dict()) # set the model to the ema model
    
    if validation is not None:
        return validation_result