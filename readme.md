# pytorch-mnist-VAE
## About VAE
VAE consists of two main components: an encoder and a decoder. The encoder takes in data points and maps them to a lower-dimensional representation, known as a latent space. The decoder then takes this latent representation and maps it back to the original data space.

The key idea of a VAE is to treat the encoder's mapping from data space to latent space as a probabilistic function, where the encoder outputs a mean and a covariance matrix that define a Gaussian distribution over the latent space. The decoder is trained to reconstruct the original data points from these Gaussian samples. The overall objective of the VAE is to maximize the likelihood of the data given this probabilistic mapping.

By treating the encoding as a probabilistic function, we can sample from the latent space and use the decoder to generate new data points that are similar to the ones in the original dataset. In this sense, the VAE serves as a generative model.

## Loss Function
The loss function for a Variational Autoencoder (VAE) is a combination of two terms: a reconstruction loss and a regularization term. The reconstruction loss measures how well the decoder is able to reconstruct the original data points from the latent space. The regularization term encourages the encoder to produce a compact and well-structured latent representation.

The reconstruction loss is typically the mean squared error (MSE) between the original data points and the reconstructed data points produced by the decoder. The MSE measures the difference between the two and provides a scalar value that the optimizer can use to update the model's parameters.

The regularization term is designed to encourage the encoder to produce a latent representation that is well-structured. In the case of a VAE, the KL divergence is used to measure the difference between the distribution over the latent space that is produced by the encoder and a prior distribution, typically a standard normal distribution. The objective of the regularization term is to minimize the KL divergence and thus produce a latent representation that is close to the prior distribution.

## Model
The model consists of an encoder and a decoder. The encoder outputs the mean and log variance of the latent vector from the input image. The decoder generates an image from the latent vector.

```
Model(
  (encoder_conv): Sequential(
    (0): Conv2d(1, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): Conv2d(32, 32, kernel_size=(5, 5), stride=(2, 2))
    (7): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): Flatten(start_dim=1, end_dim=-1)
  )
  (encoder_mean): Sequential(
    (0): Linear(in_features=128, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=16, bias=True)
  )
  (encoder_std): Sequential(
    (0): Linear(in_features=128, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=16, bias=True)
  )
  (decoder): Sequential(
    (0): Linear(in_features=16, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=128, bias=True)
    (3): Unflatten(dim=1, unflattened_size=[32, 2, 2])
    (4): ConvTranspose2d(32, 32, kernel_size=(5, 5), stride=(2, 2))
    (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): ReLU()
    (7): ConvTranspose2d(32, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (8): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU()
    (10): ConvTranspose2d(16, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (11): Sigmoid()
  )
)
```

## Dependency
Our code requires the following libraries:
* [PyTorch](https://pytorch.org/)
```
pip install PyYAML
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install seaborn
```

## Training and Evaluation
Example of configuration file for learning and evaluation
```
device: cuda
train:
  batch_size: 128
  train_epochs: 90
  loss:
    - bceloss
    - regularizationloss
  optim:
    name: adam
    learning_rate: 0.001
    others:
  lr_scheduler:
    name: steplr
    others:
      step_size: 30
      gamma: 0.1
  alpha: 0.1
  model_path: ./pretrained
  progress_path: ./train_progress
  plot_epochs: 10
model:
  architecture:
    encoder: ...
    decoder: ...
test:
  batch_size: 256
  n_components: 2
  model_path: ./pretrained/model_90.pt
  results_path: ./results
```

### Run Train
Use the code below to continue learning from the pretrained model.

```
python train.py --config './config/config.yaml'
```

### Run Evaluation
```
python test.py --config './config/config.yaml'
```

## Results
### Reconstuction Evaluation
Evaluate whether the input image is restored again through encoding and decoding. The image in the odd-numbered row is the original image, and the image in the even-numbered row is the generated image.

![reconstruction](https://user-images.githubusercontent.com/37692743/218276268-d18e79c4-0862-46ee-9b37-c6db1bc289cc.png)

### Random Generation Evaluation
Evaluate whether images are well generated from random values.

![random_generation](https://user-images.githubusercontent.com/37692743/218276277-ef602f9e-9bd0-4422-9dfc-8d006419b8ea.png)

### Distribution of latent vector
Plot encoded latent vector of input images.

![embedding](https://user-images.githubusercontent.com/37692743/218276286-aff4e839-d09d-4bd0-bdb8-88e3c50b4868.png)
