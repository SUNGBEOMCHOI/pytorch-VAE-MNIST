import numpy as np
import torch 
import torch.nn as nn
import yaml

from utils import build_network

class Model(nn.Module):
    def __init__(self, model_cfg, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        architecture_encoder = model_cfg['architecture']['encoder']
        architecture_decoder = model_cfg['architecture']['decoder']

        self.encoder_conv = build_network(architecture_encoder['conv'])
        self.encoder_mean = build_network(architecture_encoder['mean'])
        self.encoder_std = build_network(architecture_encoder['std'])

        self.decoder = build_network(architecture_decoder)

        self.mean, self.log_var, self.output = None, None, None

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        self.mean, self.log_var = self.encoding(x)
        z = self.add_noise(self.mean, self.log_var)
        output = self.decoding(z)

        return output

    def encoding(self, x):
        x = x.to(self.device)
        x = self.encoder_conv(x)
        self.mean = self.encoder_mean(x)
        self.log_var = self.encoder_std(x)

        return self.mean, self.log_var

    def decoding(self, x):
        x = x.to(self.device)
        self.output = self.decoder(x)

        return self.output

    def add_noise(self, mean, log_var):
        noise = torch.normal(0, 1, size=log_var.shape, device=self.device)
        std = torch.exp(0.5*log_var)
        return mean + std * noise
    