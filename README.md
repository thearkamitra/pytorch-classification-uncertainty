## Code Base For AML Term Project

This has been tested with Python `v3.6.8`, Torch `v1.3.1` and Torchvision `v0.4.2`. This code has two parts. The first part is the code for the evidential deep learning model to classify uncertainity which is built upon the repository provided by https://github.com/dougbrion/pytorch-classification-uncertainty. The support for datasets such as CIFAR 10 and fashion MNIST has been provided in addition to the MNIST dataset. Also, we have expanded the scope of models from only LeNet to LeNet and ResNet.

The second part is the GRADCAM.ipynb notebook which has been used in our experimentation to correlate uncertainity with the overlap of Gradcam's output with the original image. This is purely our contribution.

# Part 1

```shell
pip install -r requirements.txt
```

The are various arguments available for training and testing the network. When training or testing with uncertainty provide the `--uncertainty` argument in addition to one of the following for loss: `--mse`, `--digamma`, `--log`. The `--dataset` can be specified as `--mnist`, `--fmnist` or `--CIFAR`

```
python main.py --help

usage: main.py [-h] [--train] [--epochs EPOCHS] [--dropout] [--uncertainty]
               [--mse] [--digamma] [--log] [--test] [--examples]

optional arguments:
  -h, --help       show this help message and exit
  --train          To train the network.
  --epochs EPOCHS  Desired number of epochs.
  --dropout        Whether to use dropout or not.
  --uncertainty    Use uncertainty or not.
  --mse            Set this argument when using uncertainty. Sets loss
                   function to Expected Mean Square Error.
  --digamma        Set this argument when using uncertainty. Sets loss
                   function to Expected Cross Entropy.
  --log            Set this argument when using uncertainty. Sets loss
                   function to Negative Log of the Expected Likelihood.
  --test           To test the network.
  --dataset        Compulsory argument used to specify the dataset to be used.
  --mnist          Use MNIST dataset.
  --fmnist         Use Fashion MNIST dataset
  --CIFAR          Use CIFAR 10 dataset
  --outsample      Use out of sample image to generate the output (Here the out of sample image is of one of our teammates)
```

Example of how to train the network:

```shell
python main.py --train --dropout --uncertainty --mse --epochs 5 --dataset --mnist
```


Example of how to test the network:

```shell
python main.py --test --uncertainty --mse --dataset --mnist
```
The outputs will be available in the results folder.

# Part 2
This part consists of an Ipynb notebook and so it can be executed directed.