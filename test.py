import argparse

import yaml
import torch
import matplotlib.pyplot as plt

from models import Model
from utils import get_test_dataset, get_dataloader

def test(args, cfg):
    """
    Test trained model
    """
    ########################
    #   Get configuration  #
    ########################
    device = torch.device('cuda' if cfg['device']=='cuda' and torch.cuda.is_available() else 'cpu')
    test_cfg = cfg['test']
    batch_size = test_cfg['batch_size']
    model_cfg = cfg['model']

    ########################
    # Get pretrained model #
    ########################
    model = Model(model_cfg).to(device)
    checkpoint = torch.load(args.pretrained)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_dataset = get_test_dataset()
    test_loader = get_dataloader(test_dataset, batch_size, train=False)
    
    ########################
    #      Test model     #
    ########################
    for input, target in test_loader:
        input, target = input.to(device), target.to(device)
        output = model(input)
        break
    plt.imshow(input[6].detach().numpy().transpose(1, 2, 0), cmap='gray')
    plt.show()
    plt.imshow(output[6].detach().numpy().transpose(1, 2, 0), cmap='gray')
    plt.show()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='Path to config file')
    parser.add_argument('--pretrained', type=str, default='', help='Path to pretrained model file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    test(args, cfg)