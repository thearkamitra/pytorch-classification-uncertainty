import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import argparse
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from helpers import get_device, rotate_img, one_hot_embedding
from train import train_model
from test import rotating_image_classification, test_single_image
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from lenet import LeNet
import torchvision.transforms.functional as TF


def main():

    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true",
                            help="To train the network.")
    mode_group.add_argument("--test", action="store_true",
                            help="To test the network.")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Desired number of epochs.")
    parser.add_argument("--dropout", action="store_true",
                        help="Whether to use dropout or not.")
    parser.add_argument("--uncertainty", action="store_true",
                        help="Use uncertainty or not.")
    parser.add_argument("--dataset", action="store_true",
                        help="The dataset to use.")
    parser.add_argument("--outsample", action="store_true",
                        help="Use out of sample test image")

    uncertainty_type_group = parser.add_mutually_exclusive_group()
    uncertainty_type_group.add_argument("--mse", action="store_true",
                                        help="Set this argument when using uncertainty. Sets loss function to Expected Mean Square Error.")
    uncertainty_type_group.add_argument("--digamma", action="store_true",
                                        help="Set this argument when using uncertainty. Sets loss function to Expected Cross Entropy.")
    uncertainty_type_group.add_argument("--log", action="store_true",
                                        help="Set this argument when using uncertainty. Sets loss function to Negative Log of the Expected Likelihood.")
    
    dataset_type_group = parser.add_mutually_exclusive_group()
    dataset_type_group.add_argument("--mnist", action="store_true",
                                        help="Set this argument when using MNIST dataset")
    dataset_type_group.add_argument("--emnist", action="store_true",
                                        help="Set this argument when using EMNIST dataset")
    dataset_type_group.add_argument("--CIFAR", action="store_true",
                                        help="Set this argument when using CIFAR dataset")
    dataset_type_group.add_argument("--fmnist", action="store_true",
                                        help="Set this argument when using FMNIST dataset")
    args = parser.parse_args()

    if args.dataset:
        if args.mnist:
            from mnist import dataloaders, label_list
        elif args.CIFAR:
            from CIFAR import dataloaders, label_list
        elif args.fmnist:
            from fashionMNIST import dataloaders, label_list


    if args.train:
        num_epochs = args.epochs
        use_uncertainty = args.uncertainty
        num_classes = 10
        model = LeNet(dropout=args.dropout)

        if use_uncertainty:
            if args.digamma:
                criterion = edl_digamma_loss
            elif args.log:
                criterion = edl_log_loss
            elif args.mse:
                criterion = edl_mse_loss
            else:
                parser.error(
                    "--uncertainty requires --mse, --log or --digamma.")
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)

        exp_lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.1)

        device = get_device()
        model = model.to(device)

        model, metrics = train_model(model, dataloaders, num_classes, criterion,
                                     optimizer, scheduler=exp_lr_scheduler, num_epochs=num_epochs,
                                     device=device, uncertainty=use_uncertainty)

        state = {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if use_uncertainty:
            if args.digamma:
                torch.save(state, "./results/model_uncertainty_digamma.pt")
                print("Saved: ./results/model_uncertainty_digamma.pt")
            if args.log:
                torch.save(state, "./results/model_uncertainty_log.pt")
                print("Saved: ./results/model_uncertainty_log.pt")
            if args.mse:
                torch.save(state, "./results/model_uncertainty_mse.pt")
                print("Saved: ./results/model_uncertainty_mse.pt")

        else:
            torch.save(state, "./results/model.pt")
            print("Saved: ./results/model.pt")

    elif args.test:

        use_uncertainty = args.uncertainty
        device = get_device()
        model = LeNet()
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())

        if use_uncertainty:
            if args.digamma:
                checkpoint = torch.load(
                    "./results/model_uncertainty_digamma.pt")
            if args.log:
                checkpoint = torch.load("./results/model_uncertainty_log.pt")
            if args.mse:
                checkpoint = torch.load("./results/model_uncertainty_mse.pt")
        else:
            checkpoint = torch.load("./results/model.pt")
            
        filename = "./results/rotate.jpg"
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        model.eval()
        if args.outsample:
            img = Image.open("./data/arka.jpg").convert('L').resize((28, 28))
            img = TF.to_tensor(img)
            img.unsqueeze_(0)
        else:
            a = iter(dataloaders['test'])
            img, label = next(a)
        rotating_image_classification(
            model, img, filename, label_list, uncertainty=use_uncertainty)

        img = transforms.ToPILImage()(img[0][0])
        test_single_image(model, img, label_list, uncertainty=use_uncertainty)

if __name__ == "__main__":
    main()
