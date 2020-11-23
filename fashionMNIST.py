from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data_train = FashionMNIST("./data/FashionMNIST",
                   download=True,
                   train=True,
                   transform=transforms.Compose([
                       transforms.Resize((28, 28)),
                       transforms.ToTensor()]))

data_val = FashionMNIST("./data/FashionMNIST",
                 train=False,
                 download=True,
                 transform=transforms.Compose([
                     transforms.Resize((28, 28)),
                     transforms.ToTensor()]))

dataloader_train = DataLoader(
    data_train, batch_size=1000, shuffle=True, num_workers=0)
dataloader_val = DataLoader(data_val, batch_size=1000, num_workers=0)
dataloader_test = DataLoader(data_val, batch_size=1, num_workers=0, shuffle=True)


dataloaders = {
    "train": dataloader_train,
    "val": dataloader_val,
    "test": dataloader_test
}

label_list = data_val.class_to_idx
