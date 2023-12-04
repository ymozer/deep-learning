import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(".."))


import torch
import datasets.BDMediLeaves.BDML_dataset as BDML
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageOps, ImageEnhance

if __name__ == "__main__":
    train = BDML.BDML(split="Train")
    test = BDML.BDML(split="Test")
    validation = BDML.BDML(split="Validation")

    print("Train dataset size: ", len(train))
    print("Test dataset size: ", len(test))
    print("Validation dataset size: ", len(validation))

    print("Train dataset shape: ", train[0][0].shape)
    print("Test dataset shape: ", test[0][0].shape)
    print("Validation dataset shape: ", validation[0][0].shape)

    print("Train dataset label: ", train[0][1])
    print("Test dataset label: ", test[0][1])
    print("Validation dataset label: ", validation[0][1])

    plt.imshow(train[0][0].permute(1, 2, 0))
    title = train[0][1] + " " + str(0)
    plt.title(title)
    plt.show()

    augmented_train = BDML.BDML(split="Train", augment=True)
    print("Augmented Train dataset size: ", len(augmented_train))
    print("Augmented Train dataset shape: ", augmented_train[0][0].shape)
    print("Augmented Train dataset label: ", augmented_train[0][1])

    plt.imshow(augmented_train[0][0].permute(1, 2, 0))
    title = augmented_train[0][1] + " " + str(0)
    plt.title(title)
    plt.show()


