import torch
import torchvision
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def main():
    model = torchvision.models.vgg16(weights=None) 
    optimizer = optim.Adam(model.parameters())

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)

    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

if __name__ == "__main__":
    main()