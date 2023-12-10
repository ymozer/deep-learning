import io
import os
import sys
from tabnanny import check
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(".."))

import time
import click
import shutil  

import torch
import datasets.BDMediLeaves.BDML_dataset as BDML
import datasets.kvasir.kvasir as Kvasir
from torchvision import transforms
from torchvision.models import resnet50, densenet201, efficientnet_v2_m, EfficientNet_V2_M_Weights
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms.transforms import ToTensor

from PIL import Image, ImageOps, ImageEnhance

#torch.set_default_device("mps")
import click

def print_left_and_right_aligned(left_text, right_text):
    terminal_width = shutil.get_terminal_size().columns
    left_text_width = len(left_text)
    available_width_for_right_text = terminal_width - left_text_width 
    truncated_right_text = right_text[:available_width_for_right_text]
    click.echo(left_text, nl=False)
    remaining_space = terminal_width - len(left_text) - len(truncated_right_text)
    click.echo(" " * remaining_space, nl=False)
    click.echo(truncated_right_text, nl=False)
    click.echo("\r", nl=False)


# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 57 * 75, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * (height//4) * (width//4))
        x = nn.functional.relu(x)
        x = nn.functional.softmax(x, dim=1)
        return x
    
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        # Define the feature extraction part
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # The classifier will be defined later in the forward method
        self.classifier = None

    def forward(self, x):
        # Forward pass through the feature extraction part
        x = self.features(x)
        
        # Dynamically obtain the input features for the classifier
        x_flatten_size = x.view(x.size(0), -1).size(1)
        
        # Define the classifier if it hasn't been defined yet
        if self.classifier is None:
            self.classifier = nn.Sequential(
                nn.Linear(x_flatten_size, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, self.num_classes),
            )
        
        # Flatten the output for the classification part
        x = x.view(x.size(0), -1)
        
        # Forward pass through the classification part
        x = self.classifier(x)
        return x

def train_loop():
    epoch = 0
    loss=None
    step_count = 0
    try:
        for epoch in range(num_epochs):
            train_loss = 0
            train_acc = 0

            # Training
            model.train()

            for i, (images, labels) in enumerate(train_loader):
                step_count = i + 1
                images = images.to(device)

                # convert all labels to numbers using labels_dict
                labels = [labels_dict[label] for label in list(labels)]
                labels = torch.tensor(labels, dtype=torch.long, device=device)

                height, width = images.shape[-2], images.shape[-1]

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                train_loss += loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the accuracy
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                train_acc += correct / total

                time_elapsed = time.time() - since
                left_text = "Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Acc: {:.2f} images: {}".format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100, tuple(images.shape[2:4]))
                right_text = "Time elapsed {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)
                # time elapsed
                print_left_and_right_aligned(left_text, right_text)

            train_loss_list.append(train_loss / len(train_loader))
            train_acc_list.append(train_acc / len(train_loader))
        # save model
        torch.save(model.state_dict(), 'model.ckpt')
    except KeyboardInterrupt:
        click.echo(click.style(f"Training interrupted at {step_count}/{total_step}", fg="red"))
        click.echo(click.style("Saving model", fg="green"))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f"model.ckpt")
        click.echo(click.style("Model saved", fg="green"))
        sys.exit(0)


if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.Resize((229, 299)),
        '''
        # Augmentation
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        ''',
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((229, 299)),
        transforms.ToTensor(),
    ])

    # Set device MPS
    # Check that MPS is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')    
    # print device
    click.echo(click.style(f"Device:\t\t\t\t{device}", fg="green"))

    # Load dataset and apply transformations
    train = BDML.BDML(split="Train", transform=train_transform, augment=True)
    test = BDML.BDML(split="Test", transform=test_transform)
    validation = BDML.BDML(split="Validation", transform=test_transform)
    click.echo(click.style(f"train dataset length:\t\t{len(train)}", fg="green"))
    click.echo(click.style(f"test dataset length:\t\t{len(test)}", fg="green"))
    click.echo(click.style(f"validation dataset length:\t{len(validation)}", fg="green"))

    device = None


    # Hyperparameters
    num_epochs = 3
    batch_size = 10
    learning_rate = 0.001

    # Data loaders
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(dataset=validation, batch_size=batch_size, shuffle=False)

    '''
    # show images in trainloader according to batch sizes
    for i, (images, labels) in enumerate(train_loader):
        print("images shape:\t",images.shape)
        print("labels:\t\t",labels)
        #plot 
        fig, ax = plt.subplots(2, 5, figsize=(20, 10))
        for i in range(2):
            for j in range(5):
                ax[i, j].imshow(images[i*5+j].permute(1, 2, 0))
                ax[i, j].set_title(labels[i*5+j])
                ax[i, j].axis('off')
        plt.show()
    '''
            
    # Model
    #model = SimpleCNN(num_classes=10)
    #model = resnet50(pretrained=True)
    #model = densenet201(pretrained=True)
    #model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT, progress=True)
    model = CNN(num_classes=10)
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

    # Train the model
    total_step = len(train_loader)
    click.echo(click.style(f"Total Step:\t\t\t{total_step}", fg="green"))

    train_loss_list = []
    train_acc_list = []
    since = time.time()

    if os.path.exists("model.ckpt"):
        checkpoint = torch.load("model.ckpt")
        # Update model and optimizer with the loaded state_dicts
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        click.echo(click.style("Model loaded", fg="green"))
        model.train()
        train_loop()
    else:
        click.echo(click.style("Model not found", fg="red"))
        click.echo(click.style("Training model", fg="green"))
        train_loop()
    
    

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