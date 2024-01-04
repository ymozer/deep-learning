import torch
import torch.nn as nn
import torch.nn.functional as F


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
    super(CNN, self).__init__()
    self.cnn_layers = nn.Sequential(
        # Defining a 2D convolution layer
        nn.Conv2d(channels, 4, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Defining another 2D convolution layer
        nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

    # Adaptive linear layer
    self.adaptive_linear = AdaptiveLinearLayer(4, classes)

  def forward(self, x):
    # Forward pass
    x = self.cnn_layers(x)
    x = self.adaptive_linear(x)
    return x


class CNN2(nn.Module):
  def __init__(self, channels, classes):
    super(CNN2, self).__init__()
    self.cnn_layers = nn.Sequential(
      # Defining a 2D convolution layer
      nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      # Defining another 2D convolution layer
      nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )

    # Adaptive linear layer
    self.adaptive_linear = AdaptiveLinearLayer(32, classes)


  def forward(self, x):
    # Forward pass
    x = self.cnn_layers(x)
    x = self.adaptive_linear(x)
    return x


class BinaryClassification(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
    self.batchnorm1 = nn.BatchNorm2d(8)
    self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
    self.batchnorm2 = nn.BatchNorm2d(8)
    self.pool2 = nn.MaxPool2d(2)
    self.conv3 = nn.Conv2d(8, 32, 3, padding=1)
    self.batchnorm3 = nn.BatchNorm2d(32)
    self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
    self.batchnorm4 = nn.BatchNorm2d(32)
    self.pool4 = nn.MaxPool2d(2)
    self.conv5 = nn.Conv2d(32, 128, 3, padding=1)
    self.batchnorm5 = nn.BatchNorm2d(128)
    self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
    self.batchnorm6 = nn.BatchNorm2d(128)
    self.pool6 = nn.MaxPool2d(2)
    self.conv7 = nn.Conv2d(128, 2, 1)
    self.pool7 = nn.AvgPool2d(3)

  def forward(self, x):
    # -------------
    # INPUT
    # -------------
    x = x.view(-1, 3, 32, 32)

    # -------------
    # LAYER 1
    # -------------
    output_1 = self.conv1(x)
    output_1 = F.relu(output_1)
    output_1 = self.batchnorm1(output_1)

    # -------------
    # LAYER 2
    # -------------
    output_2 = self.conv2(output_1)
    output_2 = F.relu(output_2)
    output_2 = self.pool2(output_2)
    output_2 = self.batchnorm2(output_2)

    # -------------
    # LAYER 3
    # -------------
    output_3 = self.conv3(output_2)
    output_3 = F.relu(output_3)
    output_3 = self.batchnorm3(output_3)

    # -------------
    # LAYER 4
    # -------------
    output_4 = self.conv4(output_3)
    output_4 = F.relu(output_4)
    output_4 = self.pool4(output_4)
    output_4 = self.batchnorm4(output_4)

    # -------------
    # LAYER 5
    # -------------
    output_5 = self.conv5(output_4)
    output_5 = F.relu(output_5)
    output_5 = self.batchnorm5(output_5)

    # -------------
    # LAYER 6
    # -------------
    output_6 = self.conv6(output_5)
    output_6 = F.relu(output_6)
    output_6 = self.pool6(output_6)
    output_6 = self.batchnorm6(output_6)

    # --------------
    # OUTPUT LAYER
    # --------------
    output_7 = self.conv7(output_6)
    output_7 = self.pool7(output_7)
    output_7 = output_7.view(-1, 2)

    return F.softmax(output_7, dim=1)
