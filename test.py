import argparse
import collections

import yaml
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec

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
    model = Model(model_cfg, device).to(device)
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=device)
    else:
        checkpoint = torch.load(test_cfg['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_dataset = get_test_dataset()
    test_loader = get_dataloader(test_dataset, batch_size, train=False)
    origin_images = collections.defaultdict(list)
    generated_images = collections.defaultdict(list)
    
    ########################
    #      Test model      #
    ########################
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            # outputs = model(inputs)
            mean, _ = model.encoding(inputs)
            outputs = model.decoding(mean)
        
        for idx, target in enumerate(targets):
            origin_images[int(target)].append(inputs[idx].detach().cpu().numpy()[0])
            generated_images[int(target)].append(outputs[idx].detach().cpu().numpy()[0])
        break

    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.figure(figsize=(5, 10))
    for row in range(10):
        for column in range(10):
            origin_image = origin_images[row][column]
            generate_image = generated_images[row][column]
            plt.subplot(20, 10, 20*row+column+1)
            plt.axis('off')
            plt.imshow(origin_image, cmap='gray')
            plt.subplot(20, 10, 20*row+column+11)
            plt.axis('off')
            plt.imshow(generate_image, cmap='gray')
    plt.savefig('./result.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='Path to config file')
    parser.add_argument('--pretrained', type=str, default='', help='Path to pretrained model file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    test(args, cfg)