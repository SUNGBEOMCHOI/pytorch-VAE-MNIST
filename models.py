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

        self.mean, self.std, self.output = None, None, None

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        self.mean, self.std = self.encoding(x)
        z = self.add_noise(self.mean, self.std)
        output = self.decoding(z)

        return output

    def encoding(self, x):
        x = x.to(self.device)
        x = self.encoder_conv(x)
        self.mean = self.encoder_mean(x)
        self.std = self.encoder_std(x)

        return self.mean, self.std

    def decoding(self, x):
        x = x.to(self.device)
        self.output = self.decoder(x)

        return self.output

    def add_noise(self, mean, std):
        noise = torch.normal(0, 1, size=std.shape)
        return mean + std * noise

# if __name__ == '__main__':
#     with open('./config/config.yaml') as f:
#         cfg = yaml.safe_load(f)
#     model = Model(cfg['model'])
#     data = torch.randn((1, 1, 28, 28))
#     print(model(data))
    