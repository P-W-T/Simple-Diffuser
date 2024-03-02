from simple_diffuser_improved import *

class DoubleConv(nn.Module):
    """
    Combines two convolution operations and a residual connection with group normalization.
    In fact, each convolution operation is followed by a group normalization layer and a GeLU operation is performed
    after the group n,,ormalization of the first convolutional layer and the second convolutional layer.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, groupnorm_groups=1):
        """
        Initializes the DoubleConv module.
        args:
            - in_channels (int): Number of channels in the input
            - out_channels (int): Number of channels in the output
            - mid_channels (int): The number of channels after the first and before the secod convolutional layer. If None, it is equal to out_channels.
            - residual (bool): If True enable the residual connection
            - groupnorm_groups (int): The number of groups to be used for group normalization
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = out_channels
        self.groupnorm_groups = groupnorm_groups
        
        if mid_channels is not None:
            self.mid_channels = mid_channels
        self.residual = residual
        
        self.scale_residual = nn.Identity()
        if in_channels != out_channels and residual:
            self.scale_residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=True)
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groupnorm_groups, num_channels=self.mid_channels), #Normalize all the pixels over the channels
            nn.GELU(),
            nn.Conv2d(in_channels=self.mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groupnorm_groups, num_channels=out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        """
        Forward method for the double convolution module.
        args:
            - x (image): The input image.
        """
        if self.residual:
            return self.scale_residual(x) + self.double_conv(x)
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
                
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, residual=True)
        
        self.emb_layer = nn.Sequential(
            nn.Linear(emb_dim, in_channels), #Make the embedded time compatible with the downscaled image
            nn.SiLU()
        )
        
    def forward(self, x, t):
        """
        Forward method for the downscaling module.
        args:
            - x (image): The input image.
            - t (float tensor): Embedded time.
        """
        
        emb = self.emb_layer(t)[:,:, None, None] #change embedding dimension and add additional dimensions
        x = F.max_pool2d(x, kernel_size=2) # Downscale the image
        return  self.conv(x + emb) ##add embedding and perform convolution


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
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, residual=True)
        
        self.emb_layer = nn.Sequential(
            nn.Linear(emb_dim, in_channels), #Make the embedded time compatible with the upscaled image
            nn.SiLU()
        )
        
    def forward(self, x, skip_x, t):
        """
        Forward method for the downscaling module.
        args:
            - x (image): The input image.
            - skip_x (image): The image from the corresponding Down module)
            - t (float): Embedded time.
        """
        
        x = self.up(x) #Upscale the input
        x = torch.cat([skip_x, x], dim=1) #Add the image from the corresponding Down Module to the upscaled image        
        emb = self.emb_layer(t)[:,:, None, None] #change embedding dimension and add additional dimensions
        return  self.conv(x + emb) #Add embedding to the downsampled image and perform convolution

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
