from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import sys 
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(".."))

from utils.utils import plot_loss_and_accuracy
import PIL

# create 4 float array
a = np.random.rand(10)
b = np.random.rand(10)
c = np.random.rand(10)
d = np.random.rand(10)

image=plot_loss_and_accuracy(a, b, c, d)
image.save('test.png')



