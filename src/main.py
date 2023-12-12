import os
import sys
import time
import click
import threading

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, densenet201, efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision import transforms

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(".."))

#import datasets.kvasir.kvasir as Kvasir
import datasets.BDMediLeaves.BDML_dataset as BDML

from utils.utils import print_left_and_right_aligned, convert_seconds, TimerThread
from models.models import CNN


class TrainLoop(threading.Thread):
  def __init__(self, total_epoch, loss, device=None):
    threading.Thread.__init__(self, daemon=True)
    self.daemon = True
    self.device = device
    self.total_epoch = total_epoch
    self.loss = loss
    self.step_count = 0
    self.training_loss_list = []
    self.training_acc_list = []
    self.print_info = ""
    self.timer = TimerThread()
    click.echo(click.style("Training started", fg="green"))
    click.echo(click.style(f"Device:{device}", fg="green"))

  def __del__(self):
    print("Train loop thread deleted")

  def train_loop(self):
    try:
      for epoch in range(self.total_epoch):
        train_loss = 0
        train_acc = 0
        model.train()

        for i, (images, labels) in enumerate(train_loader):
          self.step_count = i + 1

          # convert all labels to numbers using labels_dict
          labels = [labels_dict[label] for label in list(labels)]
          labels = torch.tensor(
              labels, dtype=torch.long, device=self.device)

          # Move tensors to the configured device
          images = images.to(device)
          labels = labels.to(device)

          # Forward pass
          outputs = model(images)
          self.loss = criterion(outputs, labels)
          train_loss += self.loss.item()

          # Backward and optimize
          optimizer.zero_grad()
          self.loss.backward()
          optimizer.step()

          # Track the accuracy
          total = labels.size(0)
          _, predicted = torch.max(outputs.data, 1)
          correct = (predicted == labels).sum().item()
          train_acc += correct / total

          if os.name == 'nt':
            self.print_info = "Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Acc: {:.2f} Batch size: {}".format(
              epoch + 1, num_epochs, self.step_count, total_step, self.loss.item(), correct / total, batch_size)
          else:
            # TODO: implement this for linux or mac
            pass
          self.training_loss_list.append(train_loss / len(train_loader))
          self.training_acc_list.append(train_acc / len(train_loader))
      torch.save(model.state_dict(), 'model.ckpt')
    except KeyboardInterrupt:
        click.echo(click.style(
            f"Training interrupted at {self.step_count}/{total_step}", fg="red"))
        click.echo(click.style("Saving model", fg="green"))
    # torch.save({
    #    'epoch': epoch,
    #    'model_state_dict': model.state_dict(),
    #    'optimizer_state_dict': optimizer.state_dict(),
    #    'loss': loss,
    # }, f"model.ckpt")
    # click.echo(click.style("Model saved", fg="green"))
    # sys.exit(0)

  def run(self):
    self.train_loop()

def transform_resize_and_to_tensor(height:int=229, width: int = 299):
  return transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor(),
  ])

def set_device(device):
  if torch.cuda.is_available():
    device = torch.device('cuda:0')
  elif torch.backends.mps.is_available():
    device = torch.device('mps')
  else:
    device = torch.device('cpu')
  click.echo(click.style(f"Device:\t\t\t\t{device}", fg="green"))
  return device



if __name__ == "__main__":
  device = None

  # Hyperparameters
  num_epochs = 1
  batch_size = 32
  learning_rate = 0.001
  height = 32
  width = 32

  train_transform = transform_resize_and_to_tensor(height=height, width=width)
  test_transform = transform_resize_and_to_tensor()

  device=set_device(device)

  # Load dataset and apply transformations
  train = BDML.BDML(split="Train", transform=train_transform, augment=False, download=True)
  test = BDML.BDML(split="Test", transform=test_transform)
  validation = BDML.BDML(split="Validation", transform=test_transform)
  click.echo(click.style(f"train dataset length:\t\t{len(train)}", fg="green"))
  click.echo(click.style(f"test dataset length:\t\t{len(test)}", fg="green"))
  click.echo(click.style(f"validation dataset length:\t{len(validation)}", fg="green"))

  # Data loaders
  train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
  test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
  validation_loader = DataLoader(dataset=validation, batch_size=batch_size, shuffle=False)

  # Model Selection
  # model = SimpleCNN(num_classes=10)
  # model = resnet50(pretrained=True)
  # model = densenet201(pretrained=True)
  model = CNN(height=height, width=width, channels=3, classes=10)
  # model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT, progress=True)
  model = model.to(device)

  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  # labels dictionary
  labels_dict = {
    "Azadirachta indica":     0,
    "Calotropis gigantea":    1,
    "Centella asiatica":      2,
    "Hibiscus rosa-sinensis": 3,
    "Justicia adhatoda":      4,
    "Kalanchoe pinnata":      5,
    "Mikania micrantha":      6,
    "Ocimum tenuiflorum":     7,
    "Phyllanthus emblica":    8,
    "Terminalia arjuna":      9,
  }

  total_step = len(train_loader)
  click.echo(click.style(f"Total Step:\t\t\t{total_step}", fg="green"))

  # initialize train loop and timer thread
  train_loop = TrainLoop(total_epoch=num_epochs,loss=criterion, device=device)
  timer = TimerThread()

  if os.path.exists("model.ckpt"):
    # if model exists, load it and test it
    checkpoint = torch.load("model.ckpt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    click.echo(click.style("Model loaded", fg="green"))
    # TODO: test model

  else:
    # if model doesn't exist, train it
    click.echo(click.style("Model not found", fg="red"))
    train_loop.start()
    timer.start()
    while True:
      print_left_and_right_aligned(train_loop.print_info, f"Time: {convert_seconds(timer.seconds)}")
      time.sleep(0.1)
      if train_loop.step_count == total_step:
        break

  test_loss = 0
  test_acc = 0
  validation_loss = 0
  validation_acc = 0

  test_loss_list = []
  test_acc_list = []
  validation_loss_list = []
  validation_acc_list = []
  # Testing
  model.eval()
  with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
      images = images.to(device)
      labels = labels.to(device)
      height, width = images.shape[-2], images.shape[-1]
      # Forward pass
      outputs = model(images, height, width)
      loss = criterion(outputs, labels)
      test_loss += loss.item()
      # Track the accuracy
      total = labels.size(0)
      _, predicted = torch.max(outputs.data, 1)
      correct = (predicted == labels).sum().item()
      test_acc += correct / total

  # Validation
  model.eval()
  with torch.no_grad():
      for i, (images, labels) in enumerate(validation_loader):
          images = images.to(device)
          labels = labels.to(device)
          height, width = images.shape[-2], images.shape[-1]
          # Forward pass
          outputs = model(images, height, width)
          loss = criterion(outputs, labels)
          validation_loss += loss.item()
          # Track the accuracy
          total = labels.size(0)
          _, predicted = torch.max(outputs.data, 1)
          correct = (predicted == labels).sum().item()
          validation_acc += correct / total

  # Print the results
  test_loss_list.append(test_loss / len(test_loader))
