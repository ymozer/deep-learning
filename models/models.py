import torch
import torch.nn as nn


class AdaptiveLinearLayer(nn.Module):
  def __init__(self, input_channels, output_size):
    super(AdaptiveLinearLayer, self).__init__()

    # Global Average Pooling
    self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

    # Linear layer
    # Specify the desired output size
    self.linear = nn.Linear(input_channels, output_size)

  def forward(self, x):
    # Apply global average pooling
    x = self.global_avg_pooling(x)

    # Reshape to (batch_size, input_channels)
    x = x.view(x.size(0), -1)

    # Apply linear layer
    x = self.linear(x)

    return x

class CNN(nn.Module):
  def __init__(self, height, width, channels, classes):
    super().__init__()
    self.conv1 = nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=1, padding=1)
    self.act1 = nn.ReLU()
    self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
    self.act2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
    

  def forward(self, x):
    x = self.act1(self.conv1(x))
    x = self.drop1(x)
    x = self.act2(self.conv2(x))
    x = self.pool2(x)
    x = self.adaptive_linear(x)
    x = self.fc4(x)

    return x
