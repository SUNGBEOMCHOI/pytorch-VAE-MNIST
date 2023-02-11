import numpy as np
import torch 
import torch.nn as nn

from utils import build_network

class Model(nn.Module):
    def __init__(self, model_cfg, device=torch.device('cpu')):
        """
        VAE model

        Args:
            model_cfg: Dictionary of model configuration
            device: Torch device to use
        
        """
        super().__init__()
        self.device = device
        architecture_encoder = model_cfg['architecture']['encoder']
        architecture_decoder = model_cfg['architecture']['decoder']

        self.encoder_conv = build_network(architecture_encoder['conv'])
        self.encoder_mean = build_network(architecture_encoder['mean'])
        self.encoder_log_var = build_network(architecture_encoder['log_var'])

        self.decoder = build_network(architecture_decoder)

        self.mean, self.log_var, self.output = None, None, None

    def forward(self, x):
        """
        Encode input images, Decode created latent vector with mixed noise

        Args:
            Numpy array or torch tensor of batch images

        Returns:
            output: Torch tensor of decoded data
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        self.mean, self.log_var = self.encoding(x)
        z = self.add_noise(self.mean, self.log_var)
        output = self.decoding(z)

        return output

    def encoding(self, x):
        """
        Encode input images to gaussian distribution

        Args:
            x: Torch tensor of batch images

        Returns:
            mean: Torch tensor of mean of encoded image
            log_var: Torch tensor of log variance of encoded image        
        """
        x = x.to(self.device)
        x = self.encoder_conv(x)
        self.mean = self.encoder_mean(x)
        self.log_var = self.encoder_log_var(x)

        return self.mean, self.log_var

    def decoding(self, x):
        """
        Decode latent vectors to images

        Args:
            x: Torch tensor of batch latent vectors

        Returns:
            output: Torch tensor of decoded batch images
        """
        x = x.to(self.device)
        self.output = self.decoder(x)

        return self.output

    def add_noise(self, mean, log_var):
        """
        Get random value from Gaussian distribution which follows mean and standard deviation

        Args:
            mean: Torch tensor of mean
            log_var: Torch tensor of log variance

        Returns:
            random values from Gaussian distribution 
        """
        noise = torch.normal(0, 1, size=log_var.shape, device=self.device)
        std = torch.exp(0.5*log_var)
        return mean + std * noise
    